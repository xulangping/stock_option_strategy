#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Flow QQQ Rotation Strategy Backtest V7
Strategy: Hold 100% QQQ as base, rotate 20% positions into option flow signals
- Start with 100% QQQ
- Monitor signals after 15:45
- When signal appears for QQQ constituent, sell 20% QQQ, buy 20% into signal stock
- Stop loss: 10%, Take profit: 15%
- After exit, buy back QQQ with proceeds
- Keep QQQ position >= 50% (only trade when QQQ >= 70%)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import pytz
import warnings
import json
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Import position calculation function
from stock_filter import calculate_position_size


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    shares: int
    position_value: float
    planned_exit_date: datetime = None
    highest_price: float = None
    max_holding_days: int = 6  # Maximum holding period
    qqq_shares_sold: int = 0  # Track how many QQQ shares were sold for this rotation
    
    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price


class OptionFlowBacktestV7:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital  # Start with initial capital in cash
        self.capital = initial_capital
        
        # Positions: {"QQQ": Position, "AAPL": Position, ...}
        self.positions: Dict[str, Position] = {}
        
        self.trades_history = []
        self.daily_trades = {}
        self.stock_data_cache = {}
        
        # Market data cache: {symbol: DataFrame with datetime index}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Strategy parameters
        self.qqq_base_position = 1.0  # Start with 100% QQQ
        self.rotation_size = 0.20  # Rotate 20% per signal
        self.min_qqq_position = 0.50  # Keep QQQ >= 50%
        self.trade_threshold = 0.70  # Only trade when QQQ >= 70%
        
        self.take_profit = 0.3  # 15% take profit
        self.stop_loss = -0.05  # 10% stop loss
        
        self.trade_start_time = (10, 0)  # Signal filter time (1:00 PM)
        self.min_option_premium = 100000  # 100K threshold
        
        # Transaction costs
        self.commission_per_share = 0.005
        self.min_commission = 1.0
        self.slippage = 0.001
        
        # Data paths
        self.stock_data_dir = "stock_data_csv_min"
        self.qqq_data_file = "qqq_minute_data.csv"
        
        # Load QQQ constituents
        self.qqq_constituents = self.load_qqq_constituents()
        
    def load_qqq_constituents(self) -> set:
        """Load QQQ constituent stocks from JSON file"""
        try:
            with open('qqq_constituents.json', 'r') as f:
                data = json.load(f)
                constituents = set(data['constituents'])
                print(f"Loaded {len(constituents)} QQQ constituents")
                return constituents
        except Exception as e:
            print(f"Error loading QQQ constituents: {e}")
            return set()
    
    def calculate_commission(self, shares: int) -> float:
        """Calculate trading commission"""
        return max(shares * self.commission_per_share, self.min_commission)
    
    def convert_option_time_to_market_time(self, date_str: str, time_str: str) -> datetime:
        """Convert UTC+8 (Beijing time) to ET (New York time) and handle date adjustment"""
        option_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        
        hour = int(time_str.split(':')[0])
        if hour < 12:
            option_datetime = option_datetime + timedelta(days=1)
        
        beijing_tz = pytz.timezone('Asia/Shanghai')
        ny_tz = pytz.timezone('America/New_York')
        
        beijing_time = beijing_tz.localize(option_datetime)
        ny_time = beijing_time.astimezone(ny_tz)
        
        return ny_time.replace(tzinfo=None)
    
    def is_late_trade_time(self, signal_time: datetime) -> bool:
        """Check if signal is after trade start time (1:00 PM)"""
        return (signal_time.hour > self.trade_start_time[0] or 
                (signal_time.hour == self.trade_start_time[0] and 
                 signal_time.minute >= self.trade_start_time[1]))
    
    def load_market_data(self, symbol: str) -> bool:
        """Load and cache market data for a symbol"""
        if symbol in self.market_data:
            return True
        
        # Special handling for QQQ
        if symbol == 'QQQ':
            try:
                df = pd.read_csv(self.qqq_data_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                self.market_data['QQQ'] = df
                print(f"  Loaded {len(df)} bars for QQQ")
                return True
            except Exception as e:
                print(f"  Error loading QQQ data: {e}")
                return False
        
        # Load individual stock data
        symbol_files = []
        for filename in os.listdir(self.stock_data_dir):
            if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                symbol_files.append(filename)
        
        if not symbol_files:
            return False
        
        # Load and merge all files
        all_dfs = []
        for filename in symbol_files:
            filepath = os.path.join(self.stock_data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['timestamp'])
                all_dfs.append(df)
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        
        if not all_dfs:
            return False
        
        # Merge, deduplicate and sort
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['timestamp']).sort_values('datetime').reset_index(drop=True)
        merged_df = merged_df.set_index('datetime')
        
        self.market_data[symbol] = merged_df
        print(f"  Loaded {len(merged_df)} bars for {symbol}")
        return True
    
    def get_price_at_time(self, symbol: str, target_time: datetime) -> Optional[float]:
        """Get price for a symbol at target time (forward fill if missing)"""
        if symbol not in self.market_data:
            return None
        
        df = self.market_data[symbol]
        
        # Try to get exact time first
        if target_time in df.index:
            return df.loc[target_time]['close']
        
        # If exact time not found, use forward fill approach
        # Get the last price before or at target_time
        past_data = df[df.index <= target_time]
        if len(past_data) > 0:
            return past_data.iloc[-1]['close']
        
        # If no past data, get first future data
        future_data = df[df.index > target_time]
        if len(future_data) > 0:
            return future_data.iloc[0]['close']
        
        return None
    
    def execute_trade(self, symbol: str, trade_time: datetime, action: str, price: float, shares: int):
        """Execute and record trades"""
        commission = self.calculate_commission(shares)
        
        if action == 'BUY':
            actual_price = price * (1 + self.slippage)
        else:
            actual_price = price * (1 - self.slippage)
        
        trade = {
            'symbol': symbol,
            'time': trade_time,
            'action': action,
            'price': price,
            'actual_price': actual_price,
            'shares': shares,
            'gross_value': actual_price * shares,
            'commission': commission,
            'slippage_cost': abs(actual_price - price) * shares
        }
        
        if action in ['BUY', 'ROTATION_BUY_BACK']:
            trade['net_value'] = trade['gross_value'] + commission
            self.cash -= trade['net_value']
            
        elif action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'MAX_HOLDING_EXIT', 'ROTATION_SELL']:
            trade['net_value'] = trade['gross_value'] - commission
            self.cash += trade['net_value']
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                entry_price = pos.entry_price
                buy_commission = self.calculate_commission(shares)
                buy_slippage = entry_price * self.slippage * shares
                sell_slippage = trade['slippage_cost']
                total_commission = buy_commission + commission
                total_slippage = buy_slippage + sell_slippage
                
                trade['gross_pnl'] = (price - entry_price) * shares
                trade['net_pnl'] = trade['gross_pnl'] - total_commission - total_slippage
                trade['gross_return'] = (price - entry_price) / entry_price
                trade['net_return'] = trade['net_pnl'] / (entry_price * shares + buy_commission + buy_slippage)
        
        self.trades_history.append(trade)
    
    def get_current_portfolio_value(self, current_time: datetime) -> float:
        """Calculate current portfolio value"""
        total = self.cash
        for symbol, pos in self.positions.items():
            current_price = self.get_price_at_time(symbol, current_time)
            if current_price:
                total += pos.shares * current_price
        return total
    
    def get_qqq_weight(self, current_time: datetime) -> float:
        """Calculate current QQQ weight in portfolio"""
        if 'QQQ' not in self.positions:
            return 0.0
        
        qqq_price = self.get_price_at_time('QQQ', current_time)
        if qqq_price is None:
            return 0.0
        
        # Calculate total value of all positions (not including cash, since we're on margin)
        total_positions_value = 0
        for symbol, pos in self.positions.items():
            price = self.get_price_at_time(symbol, current_time)
            if price:
                total_positions_value += pos.shares * price
        
        if total_positions_value == 0:
            return 0.0
        
        qqq_value = self.positions['QQQ'].shares * qqq_price
        return qqq_value / total_positions_value
    
    def initialize_qqq_position(self, start_time: datetime):
        """Initialize 100% QQQ position at start"""
        if not self.load_market_data('QQQ'):
            raise Exception("Failed to load QQQ data")
        
        qqq_price = self.get_price_at_time('QQQ', start_time)
        if qqq_price is None:
            raise Exception(f"No QQQ price data at {start_time}")
        
        shares = int(self.initial_capital / (qqq_price * (1 + self.slippage)))
        
        print(f"\n[INITIAL] Buying QQQ at {start_time}")
        print(f"  Price: ${qqq_price:.2f}, Shares: {shares}")
        print(f"  Position value: ${shares * qqq_price:,.2f}")
        
        self.execute_trade('QQQ', start_time, 'BUY', qqq_price, shares)
        
        self.positions['QQQ'] = Position(
            symbol='QQQ',
            entry_time=start_time,
            entry_price=qqq_price,
            shares=shares,
            position_value=shares * qqq_price
        )
        
        print(f"  Cash after: ${self.cash:,.2f}")
    
    def check_position_exits(self, current_time: datetime):
        """Check all non-QQQ positions for exit conditions (called every minute)"""
        positions_to_close = []
        
        for symbol, pos in list(self.positions.items()):
            if symbol == 'QQQ':  # Don't check QQQ for exits
                continue
            
            if symbol not in self.market_data:
                continue
            
            current_price = self.get_price_at_time(symbol, current_time)
            if current_price is None:
                continue
            
            # Update highest price
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            
            returns = (current_price - pos.entry_price) / pos.entry_price
            holding_days = (current_time.date() - pos.entry_time.date()).days
            
            # Check maximum holding period (6 days)
            if holding_days >= pos.max_holding_days:
                positions_to_close.append((symbol, current_price, 'MAX_HOLDING_EXIT', returns))
                continue
            
            # Check stop loss (-10%)
            if returns <= self.stop_loss:
                positions_to_close.append((symbol, current_price, 'STOP_LOSS', returns))
                continue
            
            # Check take profit (+15%)
            if returns >= self.take_profit:
                positions_to_close.append((symbol, current_price, 'TAKE_PROFIT', returns))
                continue
        
        # Execute closes and buy back QQQ
        for symbol, price, reason, returns in positions_to_close:
            pos = self.positions[symbol]
            holding_days = (current_time.date() - pos.entry_time.date()).days
            print(f"\n  [{reason}] {symbol}: {pos.shares} shares @ ${price:.2f} at {current_time} (Return: {returns:.2%}, Held: {holding_days} days)")
            
            self.execute_trade(symbol, current_time, reason, price, pos.shares)
            
            # Buy back QQQ with proceeds (remember how many QQQ shares we should buy back)
            qqq_shares_sold = pos.qqq_shares_sold
            del self.positions[symbol]
            
            self.buy_back_qqq(current_time, target_shares=qqq_shares_sold)
    
    def buy_back_qqq(self, current_time: datetime, target_shares: int = None):
        """Buy back QQQ with available cash
        
        Args:
            current_time: Current time
            target_shares: Target number of shares to buy back (from rotation). 
                          If None or less than affordable, use ALL available cash.
        """
        if self.cash <= 0:
            return
        
        qqq_price = self.get_price_at_time('QQQ', current_time)
        if qqq_price is None:
            return
        
        # Calculate how many shares we can buy with available cash
        cost_per_share = qqq_price * (1 + self.slippage) + self.commission_per_share
        max_shares = int(self.cash / cost_per_share)
        
        # Strategy: Buy back all we can afford
        # The target_shares is just a sanity check
        shares_to_buy = max_shares
        
        if shares_to_buy > 0:
            print(f"  [ROTATION_BUY_BACK] Buying back QQQ: {shares_to_buy} shares @ ${qqq_price:.2f}")
            self.execute_trade('QQQ', current_time, 'ROTATION_BUY_BACK', qqq_price, shares_to_buy)
            
            if 'QQQ' in self.positions:
                # Add to existing position (update average price)
                old_pos = self.positions['QQQ']
                total_shares = old_pos.shares + shares_to_buy
                total_value = old_pos.shares * old_pos.entry_price + shares_to_buy * qqq_price
                avg_price = total_value / total_shares
                
                self.positions['QQQ'] = Position(
                    symbol='QQQ',
                    entry_time=old_pos.entry_time,
                    entry_price=avg_price,
                    shares=total_shares,
                    position_value=total_value
                )
            else:
                # Create new QQQ position
                self.positions['QQQ'] = Position(
                    symbol='QQQ',
                    entry_time=current_time,
                    entry_price=qqq_price,
                    shares=shares_to_buy,
                    position_value=shares_to_buy * qqq_price
                )
    
    def process_signal(self, signal_time: datetime, symbol: str, option_premium: float):
        """Process an option flow signal"""
        print(f"\n{'='*60}")
        print(f"Signal: {symbol} at {signal_time}, Premium: ${option_premium:,.0f}")
        
        # Check if premium meets threshold
        if option_premium < self.min_option_premium:
            print(f"  Skipping - premium below threshold")
            return
        
        # Check if symbol is in QQQ constituents
        if symbol not in self.qqq_constituents:
            print(f"  Skipping - not in QQQ constituents")
            return
        
        # Check if we already have this position
        if symbol in self.positions:
            print(f"  Skipping - already have position")
            return
        
        # Load market data first to ensure we have price data
        if not self.load_market_data(symbol):
            print(f"  Skipping - no data available")
            return
        
        # Get current prices (entry is 2 minutes after signal)
        entry_time = signal_time + timedelta(minutes=2)
        entry_price = self.get_price_at_time(symbol, entry_time)
        qqq_price = self.get_price_at_time('QQQ', entry_time)
        
        if entry_price is None or qqq_price is None:
            print(f"  Skipping - no price data")
            return
        
        # Check QQQ weight at entry time (not signal time)
        qqq_weight = self.get_qqq_weight(entry_time)
        print(f"  Current QQQ weight: {qqq_weight:.1%}")
        
        if qqq_weight < self.trade_threshold:
            print(f"  Skipping - QQQ weight {qqq_weight:.1%} < {self.trade_threshold:.1%}")
            return
        
        # Calculate rotation size (use entry_time to match qqq_price)
        portfolio_value = self.get_current_portfolio_value(entry_time)
        rotation_value = portfolio_value * self.rotation_size
        
        # Sell 20% of QQQ
        qqq_pos = self.positions.get('QQQ')
        if not qqq_pos:
            print(f"  Error - no QQQ position found")
            return
        
        print(f"  DEBUG: QQQ shares before sell: {qqq_pos.shares}, Portfolio value: ${portfolio_value:,.2f}, QQQ price: ${qqq_price:.2f}")
        print(f"  DEBUG: Rotation value (20%): ${rotation_value:,.2f}")
        
        qqq_shares_to_sell = int(rotation_value / qqq_price)
        print(f"  DEBUG: Calculated shares to sell: {qqq_shares_to_sell}")
        if qqq_shares_to_sell == 0 or qqq_shares_to_sell > qqq_pos.shares:
            print(f"  Error - invalid QQQ shares to sell")
            return
        
        print(f"\n  [ROTATION_SELL] Selling {qqq_shares_to_sell} QQQ @ ${qqq_price:.2f}")
        
        # Calculate proceeds from selling QQQ (before executing trade)
        qqq_sell_proceeds = qqq_shares_to_sell * qqq_price * (1 - self.slippage) - self.calculate_commission(qqq_shares_to_sell)
        
        self.execute_trade('QQQ', entry_time, 'ROTATION_SELL', qqq_price, qqq_shares_to_sell)
        
        # Update QQQ position
        qqq_pos.shares -= qqq_shares_to_sell
        if qqq_pos.shares == 0:
            del self.positions['QQQ']
        
        # Buy target stock using the proceeds from QQQ sale
        # Calculate maximum shares we can buy with the proceeds
        shares_to_buy = int(qqq_sell_proceeds / (entry_price * (1 + self.slippage) + self.commission_per_share))
        
        if shares_to_buy > 0:
            print(f"  [BUY] {symbol}: {shares_to_buy} shares @ ${entry_price:.2f} (using ${qqq_sell_proceeds:,.2f} from QQQ sale)")
            self.execute_trade(symbol, entry_time, 'BUY', entry_price, shares_to_buy)
            
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                shares=shares_to_buy,
                position_value=shares_to_buy * entry_price,
                qqq_shares_sold=qqq_shares_to_sell  # Remember how many QQQ shares we sold
            )
            
            print(f"  Cash after: ${self.cash:,.2f}")
        else:
            print(f"  Error - insufficient proceeds to buy any shares (proceeds: ${qqq_sell_proceeds:,.2f})")
    
    def run_backtest(self, option_flows_file: str, start_date: str = "2023-03-10", end_date: str = "2025-09-30"):
        """Run the backtest with minute-by-minute position monitoring"""
        print("Loading option flow data...")
        df = pd.read_csv(option_flows_file)
        
        # Convert times to market time
        df['market_time'] = df.apply(
            lambda row: self.convert_option_time_to_market_time(row['date'], row['time']), 
            axis=1
        )
        
        # Filter for late signals only (after 15:45)
        print(f"\nTotal signals: {len(df)}")
        df_late = df[df['market_time'].apply(self.is_late_trade_time)]
        print(f"Signals after {self.trade_start_time[0]}:{self.trade_start_time[1]:02d}: {len(df_late)}")
        
        # Filter by date range
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        df_late = df_late[(df_late['market_time'] >= start_datetime) & (df_late['market_time'] <= end_datetime)]
        print(f"Signals from {start_date} to {end_date}: {len(df_late)}")
        
        # Sort by time
        df_late = df_late.sort_values('market_time').reset_index(drop=True)
        
        # Initialize QQQ position
        first_signal_time = df_late.iloc[0]['market_time']
        init_time = first_signal_time.replace(hour=9, minute=30, second=0)
        self.initialize_qqq_position(init_time)
        
        # Create signal lookup dictionary for fast access
        signals_by_time = {}
        for idx, row in df_late.iterrows():
            signal_time = row['market_time']
            if signal_time not in signals_by_time:
                signals_by_time[signal_time] = []
            signals_by_time[signal_time].append({
                'symbol': row['underlying_symbol'],
                'premium': row['premium']
            })
        
        print(f"\nRunning backtest with minute-by-minute monitoring...")
        print(f"Will check positions every minute during market hours (9:30-16:00)")
        print(f"Total signals to process: {len(signals_by_time)}")
        
        # Get all unique dates from QQQ data
        if not self.load_market_data('QQQ'):
            raise Exception("Failed to load QQQ data")
        
        qqq_df = self.market_data['QQQ']
        
        # Filter to backtest date range
        qqq_df_filtered = qqq_df[(qqq_df.index >= start_datetime) & (qqq_df.index <= end_datetime)]
        
        print(f"Processing {len(qqq_df_filtered)} minute bars...")
        
        # Count signals by year for debugging
        signal_years = {}
        for sig_time in signals_by_time.keys():
            year = sig_time.year
            signal_years[year] = signal_years.get(year, 0) + len(signals_by_time[sig_time])
        print(f"Signals by year: {signal_years}")
        
        minute_count = 0
        last_date = None
        
        # Process each minute in the market
        for current_time in qqq_df_filtered.index:
            minute_count += 1
            
            # Print progress every trading day
            if current_time.date() != last_date:
                last_date = current_time.date()
                if minute_count > 1:
                    print(f"  Processing {current_time.date()}...")
            
            # Check positions for exits at every minute
            if len(self.positions) > 1:  # If we have non-QQQ positions
                self.check_position_exits(current_time)
            
            # Process signals at this time (if any)
            if current_time in signals_by_time:
                for signal in signals_by_time[current_time]:
                    self.process_signal(
                        signal_time=current_time,
                        symbol=signal['symbol'],
                        option_premium=signal['premium']
                    )
        
        print(f"\n{'='*60}")
        print(f"Backtest completed! Processed {minute_count} minute bars")
        print(f"{'='*60}")
    
    def calculate_metrics(self):
        """Calculate and display backtest metrics"""
        # Calculate total P&L
        total_gross_pnl = sum(trade.get('gross_pnl', 0) for trade in self.trades_history)
        total_net_pnl = sum(trade.get('net_pnl', 0) for trade in self.trades_history)
        total_commission = sum(trade.get('commission', 0) for trade in self.trades_history)
        total_slippage = sum(trade.get('slippage_cost', 0) for trade in self.trades_history)
        
        # Calculate final capital (need to mark to market QQQ position)
        # Use the last time point in QQQ data for final valuation
        if 'QQQ' in self.market_data and len(self.market_data['QQQ']) > 0:
            last_time = self.market_data['QQQ'].index[-1]
            final_value = self.get_current_portfolio_value(last_time)
        else:
            # Fallback to entry price based calculation
            open_position_value = sum(pos.shares * pos.entry_price for pos in self.positions.values())
            final_value = self.cash + open_position_value
        
        self.capital = final_value
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Get completed trades (exits only)
        completed_trades = [t for t in self.trades_history if 'net_pnl' in t]
        winning_trades = [t for t in completed_trades if t['net_pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['net_pnl'] <= 0]
        
        # Calculate metrics
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        avg_return = np.mean([t['net_return'] for t in completed_trades]) if completed_trades else 0
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS - V7 QQQ ROTATION STRATEGY")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        
        print(f"\nTotal Trades: {len(self.trades_history)}")
        print(f"Completed Round Trips: {len(completed_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Average Return per Trade: {avg_return:.2%}")
        
        print(f"\nGross P&L: ${total_gross_pnl:,.2f}")
        print(f"Total Commission: ${total_commission:,.2f}")
        print(f"Total Slippage: ${total_slippage:,.2f}")
        print(f"Net P&L: ${total_net_pnl:,.2f}")
        
        # Trade breakdown
        buy_count = len([t for t in self.trades_history if t['action'] == 'BUY'])
        rotation_sell_count = len([t for t in self.trades_history if t['action'] == 'ROTATION_SELL'])
        rotation_buyback_count = len([t for t in self.trades_history if t['action'] == 'ROTATION_BUY_BACK'])
        stop_loss_count = len([t for t in self.trades_history if t['action'] == 'STOP_LOSS'])
        take_profit_count = len([t for t in self.trades_history if t['action'] == 'TAKE_PROFIT'])
        max_holding_count = len([t for t in self.trades_history if t['action'] == 'MAX_HOLDING_EXIT'])
        
        print(f"\nTrade Breakdown:")
        print(f"  Stock Buys: {buy_count}")
        print(f"  QQQ Rotation Sells: {rotation_sell_count}")
        print(f"  QQQ Buy Backs: {rotation_buyback_count}")
        print(f"  Stop Losses: {stop_loss_count}")
        print(f"  Take Profits: {take_profit_count}")
        print(f"  Max Holding Exits (6 days): {max_holding_count}")
        
        # Show open positions
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos.shares} shares @ ${pos.entry_price:.2f} (Value: ${pos.shares * pos.entry_price:,.2f})")
        
        # Save detailed trade log
        if self.trades_history:
            trade_df = pd.DataFrame(self.trades_history)
            trade_df = trade_df.sort_values('time').reset_index(drop=True)
            trade_df.to_csv('backtest_trades_v7_qqq_rotation.csv', index=False)
            print(f"\nDetailed trade log saved to 'backtest_trades_v7_qqq_rotation.csv'")


def main():
    print("\nInitializing V7 QQQ Rotation Strategy Backtest:")
    print(f"  Initial capital: $100,000")
    print(f"  Base position: 100% QQQ")
    print(f"  Rotation size: 20% per signal")
    print(f"  Min QQQ position: 50%")
    print(f"  Trade threshold: QQQ >= 70%")
    print(f"  Take profit: 15%")
    print(f"  Stop loss: 10%")
    print(f"  Max holding days: 6 days")
    print(f"  Signal filter: After 1:00 PM")
    print(f"  Position monitoring: Every minute\n")
    
    backtest = OptionFlowBacktestV7(initial_capital=100000)
    
    option_file = "option_data/merged_deduplicated_2023M3_2025M9_all.csv"
    
    backtest.run_backtest(option_file, start_date="2023-03-10", end_date="2025-09-30")
    backtest.calculate_metrics()


if __name__ == "__main__":
    main()

