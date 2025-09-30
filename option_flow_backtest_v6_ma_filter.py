#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Flow Momentum Strategy Backtest V6 - Event-Driven with MA Filter
Event-driven architecture with MA bullish filter (MA5 > MA10 > MA20)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import pytz
import warnings
import re
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Import stock filter functions
from stock_filter import read_all_parquet_files, MA_Bullish_Signal


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    shares: int
    option_premium: float
    planned_exit_date: datetime
    highest_price: float = None
    
    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price


class Event:
    """Base event class with time for priority queue"""
    def __init__(self, time: datetime, priority: int = 0):
        self.time = time
        self.priority = priority
    
    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time


class SignalEvent(Event):
    """Option flow signal event"""
    def __init__(self, signal_time: datetime, symbol: str, option_premium: float, 
                 option_chain_id: str, priority: int = 1):
        super().__init__(time=signal_time, priority=priority)
        self.signal_time = signal_time
        self.symbol = symbol
        self.option_premium = option_premium
        self.option_chain_id = option_chain_id


class OptionFlowBacktestV6MA:
    def __init__(self, initial_capital: float = 100000, holding_days: int = 1):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades_history = []
        self.daily_trades = {}
        self.stock_blacklist = {}
        self.stock_data_cache = {}
        
        # Market data cache: {symbol: DataFrame with datetime index}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Filter statistics
        self.total_signals = 0
        self.ma_filtered_signals = 0
        self.premium_filtered_signals = 0
        self.dte_filtered_signals = 0
        
        # Load daily data for MA filtering
        print("Loading daily data for MA filtering...")
        self.daily_data = read_all_parquet_files("daily/")
        if self.daily_data.empty:
            print("Warning: No daily data loaded for MA filtering")
        else:
            print(f"Loaded {len(self.daily_data)} rows of daily data for MA filtering")
        
        # Strategy parameters
        self.max_daily_trades = 4
        self.max_daily_position = 0.80
        self.min_cash_ratio = 0.20
        self.take_profit = 0.25
        self.stop_loss = -0.1
        self.holding_days = holding_days
        self.blacklist_days = 5
        self.entry_delay = 5
        self.trade_start_time = (13, 00)
        
        # Filter parameters
        self.min_option_premium = 100000  # 100K threshold
        self.max_dte = 100  # Maximum days to expiration
        
        # Stocks to skip
        self.skip_stocks = {}
        
        # Transaction costs
        self.commission_per_share = 0.005
        self.min_commission = 1.0
        self.slippage = 0.001
        
        # Data paths
        self.stock_data_dir = "stock_data_csv_min"
        
    def calculate_commission(self, shares: int) -> float:
        """Calculate trading commission"""
        return max(shares * self.commission_per_share, self.min_commission)
    
    def extract_expiration_from_option_id(self, option_chain_id: str) -> Optional[datetime]:
        """Extract expiration date from option chain ID
        Format: GE230421C00100000 -> 2023-04-21
        """
        match = re.search(r'[A-Z]+([0-9]{6})[CP]', option_chain_id)
        if match:
            date_str = match.group(1)
            year = int('20' + date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return datetime(year, month, day)
        return None
        
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
        """Check if signal is after trade start time"""
        return (signal_time.hour > self.trade_start_time[0] or 
                (signal_time.hour == self.trade_start_time[0] and 
                 signal_time.minute >= self.trade_start_time[1]))
    
    def calculate_exit_date(self, entry_time: datetime) -> datetime:
        """Calculate the exit date based on holding_days parameter"""
        exit_date = entry_time.replace(hour=15, minute=0, second=0, microsecond=0)
        exit_date = exit_date + timedelta(days=self.holding_days)
        
        # Skip weekends
        while exit_date.weekday() >= 5:
            exit_date += timedelta(days=1)
            
        return exit_date
    
    def load_market_data(self, symbol: str) -> bool:
        """Load and cache market data for a symbol"""
        if symbol in self.market_data:
            return True
            
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
        """Get price for a symbol at or after target time"""
        if symbol not in self.market_data:
            return None
        
        df = self.market_data[symbol]
        future_data = df[df.index >= target_time]
        if len(future_data) == 0:
            return None
        
        return future_data.iloc[0]['close']
    
    def execute_trade(self, symbol: str, trade_time: datetime, action: str, price: float, shares: int = 0):
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
        
        if action == 'BUY':
            trade['net_value'] = trade['gross_value'] + commission
            self.cash -= trade['net_value']
            
            trade_date = trade_time.date()
            if trade_date not in self.daily_trades:
                self.daily_trades[trade_date] = 0
            self.daily_trades[trade_date] += 1
            
        elif action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'HOLDING_PERIOD_EXIT']:
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
    
    def can_open_position(self, date: datetime.date, symbol: str) -> bool:
        """Check if we can open new position"""
        # Check if already have position
        if symbol in self.positions:
            return False
        
        # Check daily trade limit
        if date in self.daily_trades and self.daily_trades[date] >= self.max_daily_trades:
            return False
        
        # Check cash
        if self.cash <= 0:
            return False
        
        # Check cash ratio
        total_portfolio_value = self.cash + sum(pos.shares * pos.entry_price for pos in self.positions.values())
        min_cash_required = total_portfolio_value * self.min_cash_ratio
        if self.cash < min_cash_required:
            return False
        
        # Check blacklist
        if symbol in self.stock_blacklist and date <= self.stock_blacklist[symbol]:
            return False
        
        return True
    
    def check_position_exits(self, current_time: datetime):
        """Check all positions for exit conditions"""
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in self.market_data:
                continue
            
            current_price = self.get_price_at_time(symbol, current_time)
            if current_price is None:
                continue
            
            # Update highest price
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            
            returns = (current_price - pos.entry_price) / pos.entry_price
            
            # Check stop loss
            if returns <= self.stop_loss:
                positions_to_close.append((symbol, current_price, 'STOP_LOSS', returns))
                continue
            
            # Check take profit
            if returns >= self.take_profit:
                positions_to_close.append((symbol, current_price, 'TAKE_PROFIT', returns))
                continue
            
            # Check holding period exit
            if (current_time.date() >= pos.planned_exit_date.date() and 
                current_time.hour >= 15):
                positions_to_close.append((symbol, current_price, 'HOLDING_PERIOD_EXIT', returns))
                continue
        
        # Execute closes
        for symbol, price, reason, returns in positions_to_close:
            pos = self.positions[symbol]
            actual_holding_days = (current_time.date() - pos.entry_time.date()).days
            print(f"  [{reason}] {symbol}: {pos.shares} shares @ ${price:.2f} at {current_time} (Return: {returns:.2%}, Held: {actual_holding_days} days)")
            self.execute_trade(symbol, current_time, reason, price, pos.shares)
            del self.positions[symbol]
    
    def process_signal(self, signal: SignalEvent):
        """Process an option flow signal"""
        symbol = signal.symbol
        signal_time = signal.signal_time
        option_premium = signal.option_premium
        option_chain_id = signal.option_chain_id
        
        print(f"\n{'='*60}")
        print(f"Signal: {symbol} at {signal_time}")
        
        # Track total signals
        self.total_signals += 1
        
        # Skip if in skip list
        if symbol in self.skip_stocks:
            print(f"  [SKIPPED] In skip list")
            return
        
        # MA Filter Check - only trade stocks with bullish MA alignment (MA5 > MA10 > MA20)
        signal_date_str = signal_time.strftime('%Y-%m-%d')
        ma_bullish = MA_Bullish_Signal(symbol, signal_date_str, self.daily_data)
        if not ma_bullish:
            print(f"  Skipping - {symbol} does not meet MA bullish criteria (MA5 > MA10 > MA20)")
            self.ma_filtered_signals += 1
            return
        else:
            print(f"  ✓ {symbol} meets MA bullish criteria on {signal_date_str}")
        
        # Filter by minimum option premium
        if option_premium < self.min_option_premium:
            print(f"  Skipping - option premium ${option_premium:,.0f} below threshold ${self.min_option_premium:,.0f}")
            self.premium_filtered_signals += 1
            return
        
        # Calculate DTE from option_chain_id
        expiration_date = self.extract_expiration_from_option_id(option_chain_id)
        if expiration_date is None:
            print(f"  Skipping - cannot parse expiration date from {option_chain_id}")
            return
        
        signal_date = signal_time.date()
        dte = (expiration_date.date() - signal_date).days
        
        # Filter by DTE
        if dte > self.max_dte:
            print(f"  Skipping - DTE {dte} exceeds maximum {self.max_dte} days")
            self.dte_filtered_signals += 1
            return
        
        print(f"  Option premium: ${option_premium:,.0f} (threshold: ${self.min_option_premium:,.0f}), DTE: {dte} (max: {self.max_dte})")
        
        # Skip if too close to market close
        if signal_time.hour == 15 and signal_time.minute >= 54:
            print(f"  Skipping - too close to market close")
            return
        
        # Check if can open position
        if not self.can_open_position(signal_time.date(), symbol):
            print(f"  Skipping - cannot open position")
            return
        
        # Load market data if not already loaded
        if not self.load_market_data(symbol):
            print(f"  Skipping - no data available")
            return
        
        # Calculate entry time
        entry_time_target = signal_time + timedelta(minutes=self.entry_delay)
        
        # Get entry price
        entry_price = self.get_price_at_time(symbol, entry_time_target)
        if entry_price is None:
            print(f"  Skipping - no price data at entry time")
            return
        
        # Find actual entry time
        df = self.market_data[symbol]
        entry_data = df[df.index >= entry_time_target]
        if len(entry_data) == 0:
            return
        entry_time = entry_data.index[0]
        
        # Calculate position size
        position_pct_of_initial = min(option_premium / 800000, 0.4)
        
        # Check daily position limit
        todays_allocation = sum(trade['gross_value'] for trade in self.trades_history 
                               if trade['time'].date() == signal_time.date() and trade['action'] == 'BUY')
        current_daily_position = todays_allocation / self.initial_capital
        
        if current_daily_position >= self.max_daily_position:
            print(f"  Daily position limit reached ({current_daily_position:.1%})")
            return
        
        max_remaining_pct = self.max_daily_position - current_daily_position
        if position_pct_of_initial > max_remaining_pct:
            position_pct_of_initial = max_remaining_pct
        
        # Calculate shares
        position_value = self.initial_capital * position_pct_of_initial
        if position_value > self.cash:
            position_value = self.cash * 0.95
        
        shares = int(position_value / entry_price)
        if shares == 0:
            print(f"  Position too small")
            return
        
        # Double-check cash
        total_cost = shares * entry_price * (1 + self.slippage) + self.calculate_commission(shares)
        if total_cost > self.cash:
            shares = int((self.cash * 0.95) / (entry_price * (1 + self.slippage) + self.commission_per_share))
            if shares <= 0:
                return
        
        # Calculate planned exit
        planned_exit_date = self.calculate_exit_date(entry_time)
        
        # Execute buy
        actual_buy_price = entry_price * (1 + self.slippage)
        actual_position_value = shares * entry_price
        actual_position_pct = actual_position_value / self.initial_capital
        
        print(f"  [BUY] {symbol}: {shares} shares @ ${entry_price:.2f} (actual: ${actual_buy_price:.2f}) at {entry_time}")
        print(f"       Position: {actual_position_pct:.1%}, Cost: ${total_cost:,.2f}, Cash before: ${self.cash:,.2f}")
        
        self.execute_trade(symbol, entry_time, 'BUY', entry_price, shares)
        
        # Create position object
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            shares=shares,
            option_premium=option_premium,
            planned_exit_date=planned_exit_date
        )
        
        # Add to blacklist
        blacklist_until = entry_time.date() + timedelta(days=self.blacklist_days)
        self.stock_blacklist[symbol] = blacklist_until
        
        print(f"       Cash after: ${self.cash:,.2f}, Planned exit: {planned_exit_date}")
    
    def run_backtest(self, option_flows_file: str):
        """Run the event-driven backtest"""
        print("Loading option flow data...")
        df = pd.read_csv(option_flows_file)
        
        # Convert times to market time
        df['market_time'] = df.apply(
            lambda row: self.convert_option_time_to_market_time(row['date'], row['time']), 
            axis=1
        )
        
        # Filter for late signals only
        print(f"\nTotal signals: {len(df)}")
        df_late = df[df['market_time'].apply(self.is_late_trade_time)]
        print(f"Signals after {self.trade_start_time[0]}:{self.trade_start_time[1]:02d} PM: {len(df_late)}")
        
        # Create signal events
        signal_events = []
        for idx, row in df_late.iterrows():
            event = SignalEvent(
                signal_time=row['market_time'],
                symbol=row['underlying_symbol'],
                option_premium=row['premium'],
                option_chain_id=row['option_chain_id']
            )
            signal_events.append(event)
        
        # Sort signals by time
        signal_events.sort()
        
        print(f"\nProcessing {len(signal_events)} signals in chronological order...")
        print(f"Initial cash: ${self.cash:,.2f}\n")
        
        # Process events
        for signal in signal_events:
            # Before processing signal, check all positions for exits at this time
            self.check_position_exits(signal.time)
            
            # Process the signal
            self.process_signal(signal)
        
        print(f"\n{'='*60}")
        print("Backtest completed!")
    
    def calculate_metrics(self):
        """Calculate and display backtest metrics"""
        # Calculate total P&L
        total_gross_pnl = sum(trade.get('gross_pnl', 0) for trade in self.trades_history)
        total_net_pnl = sum(trade.get('net_pnl', 0) for trade in self.trades_history)
        total_commission = sum(trade.get('commission', 0) for trade in self.trades_history)
        total_slippage = sum(trade.get('slippage_cost', 0) for trade in self.trades_history)
        
        # Calculate final capital
        open_position_value = sum(pos.shares * pos.entry_price for pos in self.positions.values())
        self.capital = self.cash + open_position_value
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Get winning and losing trades
        completed_trades = [t for t in self.trades_history if 'net_pnl' in t]
        winning_trades = [t for t in completed_trades if t['net_pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['net_pnl'] <= 0]
        
        # Calculate metrics
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        avg_return = np.mean([t['net_return'] for t in completed_trades]) if completed_trades else 0
        
        # Calculate daily statistics
        trades_by_date = {}
        for trade in completed_trades:
            date = trade['time'].date()
            if date not in trades_by_date:
                trades_by_date[date] = []
            trades_by_date[date].append(trade)
        
        trading_days = len(trades_by_date)
        avg_trades_per_day = len(completed_trades) / trading_days if trading_days > 0 else 0
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS - V6 EVENT-DRIVEN WITH MA FILTER")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Cash: ${self.cash:,.2f}")
        print(f"Open Positions Value: ${open_position_value:,.2f}")
        print(f"Final Capital (Cash + Positions): ${self.capital:,.2f}")
        print(f"Gross P&L: ${total_gross_pnl:,.2f}")
        print(f"Total Commission: ${total_commission:,.2f}")
        print(f"Total Slippage: ${total_slippage:,.2f}")
        print(f"Net P&L: ${total_net_pnl:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        
        print(f"\nFilter Statistics:")
        print(f"Total Signals Processed: {self.total_signals}")
        print(f"Signals Filtered by MA: {self.ma_filtered_signals}")
        print(f"Signals Filtered by Premium (<${self.min_option_premium:,.0f}): {self.premium_filtered_signals}")
        print(f"Signals Filtered by DTE (>{self.max_dte} days): {self.dte_filtered_signals}")
        if self.total_signals > 0:
            print(f"MA Filter Pass Rate: {(self.total_signals - self.ma_filtered_signals) / self.total_signals:.2%}")
            print(f"Premium Filter Pass Rate: {(self.total_signals - self.premium_filtered_signals) / self.total_signals:.2%}")
            print(f"DTE Filter Pass Rate: {(self.total_signals - self.dte_filtered_signals) / self.total_signals:.2%}")
        
        print(f"\nTotal Trades: {len(completed_trades)}")
        print(f"Trading Days: {trading_days}")
        print(f"Avg Trades per Day: {avg_trades_per_day:.2f}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Average Return per Trade: {avg_return:.2%}")
        
        # Trade breakdown by exit type
        exit_types = {}
        for trade in self.trades_history:
            if trade['action'] in ['STOP_LOSS', 'TAKE_PROFIT', 'HOLDING_PERIOD_EXIT']:
                exit_types[trade['action']] = exit_types.get(trade['action'], 0) + 1
        
        print("\nExit Type Breakdown:")
        for exit_type, count in exit_types.items():
            print(f"  {exit_type}: {count}")
        
        print(f"\nHolding Period Configuration:")
        print(f"  Configured holding days: {self.holding_days}")
        print(f"  Blacklist days: {self.blacklist_days}")
        
        # Show open positions
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos.shares} shares @ ${pos.entry_price:.2f} (Value: ${pos.shares * pos.entry_price:,.2f})")
        
        # Save detailed trade log
        if self.trades_history:
            trade_df = pd.DataFrame(self.trades_history)
            trade_df = trade_df.sort_values('time').reset_index(drop=True)
            trade_df.to_csv('backtest_trades_v6_ma_filter.csv', index=False)
            print(f"\nDetailed trade log saved to 'backtest_trades_v6_ma_filter.csv'")


def main():
    import sys
    
    # Get holding days from command line
    holding_days = 1
    if len(sys.argv) > 1:
        try:
            holding_days = int(sys.argv[1])
            print(f"Using holding_days = {holding_days}")
        except ValueError:
            print(f"Invalid holding_days argument, using default of 1")
    
    print(f"\nInitializing V6 Event-Driven Backtest with MA Filter:")
    print(f"  Initial capital: $100,000")
    print(f"  Holding days: {holding_days}")
    print(f"  Max daily position: 80%")
    print(f"  Min cash ratio: 20%")
    print(f"  Take profit: 25%")
    print(f"  Stop loss: -10%")
    print(f"  MA Filter: MA5 > MA10 > MA20")
    print(f"  Min option premium: $100,000")
    print(f"  Max DTE: 100 days\n")
    
    backtest = OptionFlowBacktestV6MA(initial_capital=100000, holding_days=holding_days)
    
    option_file = "option_data/merged_deduplicated_2023M3_2025M9_all.csv"
    
    backtest.run_backtest(option_file)
    backtest.calculate_metrics()


if __name__ == "__main__":
    main()
