#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Flow Momentum Strategy Backtest V6 - Event-Driven Architecture
Completely refactored to use event-driven approach to prevent duplicate positions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import pytz
import warnings
import heapq
import re
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
                 option_chain_id: str = '', priority: int = 1):
        super().__init__(time=signal_time, priority=priority)
        self.signal_time = signal_time
        self.symbol = symbol
        self.option_premium = option_premium
        self.option_chain_id = option_chain_id


class MarketUpdateEvent(Event):
    """Market price update event"""
    def __init__(self, time: datetime, priority: int = 0):
        super().__init__(time=time, priority=priority)


class OptionFlowBacktestV6:
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
        
        # Strategy parameters  
        self.max_daily_trades = 5
        self.max_daily_position = 0.99
        self.min_cash_ratio = -0.5  # Allow leverage: cash can go to -50% of total assets
        self.max_leverage = 1.45  # Max 1.45x leverage (留5%buffer应对价格波动)
        self.take_profit = 0.2
        self.stop_loss = -0.05
        self.holding_days = holding_days
        self.blacklist_days = 15
        self.entry_delay = 2
        self.trade_start_time = (15, 30)
        self.min_option_premium = 100000  # 100K threshold
        self.max_single_position = 0.3
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
        # Find first bar at or after target time
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
        """Check if we can open new position (with leverage support)"""
        # Check if already have position
        if symbol in self.positions:
            return False
        
        # Check daily trade limit
        if date in self.daily_trades and self.daily_trades[date] >= self.max_daily_trades:
            return False
        
        # Check leverage limit (cash ratio can be negative)
        total_portfolio_value = self.cash + sum(pos.shares * pos.entry_price for pos in self.positions.values())
        cash_ratio = self.cash / total_portfolio_value if total_portfolio_value > 0 else 0
        
        # Allow negative cash down to min_cash_ratio (e.g., -50%)
        if cash_ratio < self.min_cash_ratio:
            print(f"  Skipping - leverage limit reached (cash ratio {cash_ratio:.1%} < {self.min_cash_ratio:.1%})")
            return False
        
        # Check blacklist
        if symbol in self.stock_blacklist and date <= self.stock_blacklist[symbol]:
            return False
        
        return True
    
    def check_position_exits(self, current_time: datetime):
        """Check all positions for exit conditions
        
        IMPORTANT: This checks positions at current_time, but also closes any positions
        that should have been closed before current_time (in case of missing signals)
        """
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in self.market_data:
                continue
            
            # Check if position should have been closed before current_time
            # This handles the case where there were no signals between entry and planned exit
            exit_time = current_time
            
            # If current_time is past the planned exit date, use the planned exit date
            if current_time.date() > pos.planned_exit_date.date():
                # Position should have been closed on planned_exit_date
                # Find the price at planned_exit_date 15:00
                exit_time = pos.planned_exit_date
                current_price = self.get_price_at_time(symbol, exit_time)
                if current_price is None:
                    # If no data at exact time, use current_time price as fallback
                    exit_time = current_time
                    current_price = self.get_price_at_time(symbol, current_time)
                    if current_price is None:
                        continue
            else:
                # Normal case: check at current_time
                current_price = self.get_price_at_time(symbol, current_time)
                if current_price is None:
                    continue
            
            # Update highest price (scan all prices from entry to exit_time)
            df = self.market_data[symbol]
            price_data = df[(df.index >= pos.entry_time) & (df.index <= exit_time)]
            if len(price_data) > 0:
                max_price = price_data['close'].max()
                if max_price > pos.highest_price:
                    pos.highest_price = max_price
            
            returns = (current_price - pos.entry_price) / pos.entry_price
            
            # Check stop loss
            if returns <= self.stop_loss:
                positions_to_close.append((symbol, exit_time, current_price, 'STOP_LOSS', returns))
                continue
            
            # Check take profit
            if returns >= self.take_profit:
                positions_to_close.append((symbol, exit_time, current_price, 'TAKE_PROFIT', returns))
                continue
            
            # Check holding period exit
            if (current_time.date() >= pos.planned_exit_date.date() and 
                current_time.hour >= 15):
                positions_to_close.append((symbol, exit_time, current_price, 'HOLDING_PERIOD_EXIT', returns))
                continue
        
        # Execute closes
        for symbol, exit_time, price, reason, returns in positions_to_close:
            pos = self.positions[symbol]
            actual_holding_days = (exit_time.date() - pos.entry_time.date()).days
            print(f"  [{reason}] {symbol}: {pos.shares} shares @ ${price:.2f} at {exit_time} (Return: {returns:.2%}, Held: {actual_holding_days} days)")
            self.execute_trade(symbol, exit_time, reason, price, pos.shares)
            del self.positions[symbol]
    
    def process_signal(self, signal: SignalEvent):
        """Process an option flow signal"""
        symbol = signal.symbol
        signal_time = signal.signal_time
        option_premium = signal.option_premium
        option_chain_id = signal.option_chain_id
        
        print(f"\n{'='*60}")
        print(f"Signal: {symbol} at {signal_time}")
        
        # Calculate DTE from option_chain_id
        dte = 0  # Default DTE
        if option_chain_id:
            expiration_date = self.extract_expiration_from_option_id(option_chain_id)
            if expiration_date:
                signal_date = signal_time.date()
                dte = (expiration_date.date() - signal_date).days
        
        print(f"  Option premium: ${option_premium:,.0f}, DTE: {dte}")
        
        if option_premium < self.min_option_premium:
            print(f"  Skipping - option premium ${option_premium:,.0f} below threshold ${self.min_option_premium:,.0f}")
            return
        
        # Skip if in skip list
        if symbol in self.skip_stocks:
            print(f"  [SKIPPED] In skip list")
            return
        
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
        
        # Calculate entry time (signal + delay)
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
        
        # Calculate current total portfolio value
        current_total_assets = self.cash + sum(pos.shares * pos.entry_price for pos in self.positions.values())
        
        # Calculate position size using the unified function (as % of current assets)
        position_pct = calculate_position_size(
            option_premium=option_premium,
            dte=dte,                   # Pass actual DTE
            premium_divisor=800000,   # 2M divisor
            max_position_pct=self.max_single_position,     
            dte_weight=False           # Don't adjust by DTE in basic version
        )
        
        # Check daily position limit based on current total assets
        todays_allocation = sum(trade['gross_value'] for trade in self.trades_history 
                               if trade['time'].date() == signal_time.date() and trade['action'] == 'BUY')
        current_daily_position = todays_allocation / current_total_assets if current_total_assets > 0 else 0
        
        if current_daily_position >= self.max_daily_position:
            print(f"  Daily position limit reached ({current_daily_position:.1%} of current assets)")
            return
        
        max_remaining_pct = self.max_daily_position - current_daily_position
        if position_pct > max_remaining_pct:
            position_pct = max_remaining_pct
            print(f"  Adjusted position to {position_pct:.1%} to stay within daily limit")
        
        # Calculate position value in dollars (based on current total assets)
        position_value = current_total_assets * position_pct
        
        # With leverage enabled, we don't limit by cash
        # Instead, check if total leverage would exceed limit
        total_cost = position_value * (1 + self.slippage) + self.calculate_commission(int(position_value / entry_price))
        
        # Check if this would exceed leverage limit
        cash_after_buy = self.cash - total_cost
        positions_value_after = sum(pos.shares * pos.entry_price for pos in self.positions.values()) + position_value
        total_assets_after = cash_after_buy + positions_value_after
        
        # Calculate leverage ratio (positions / total_assets)
        leverage_ratio = positions_value_after / total_assets_after if total_assets_after > 0 else 0
        
        if leverage_ratio > self.max_leverage:
            print(f"  Skipping - would exceed max leverage ({leverage_ratio:.1%} > {self.max_leverage:.0%})")
            return
        
        shares = int(position_value / entry_price)
        if shares == 0:
            print(f"  Position too small")
            return
        
        # Calculate planned exit
        planned_exit_date = self.calculate_exit_date(entry_time)
        
        # Execute buy
        actual_buy_price = entry_price * (1 + self.slippage)
        actual_position_value = shares * entry_price
        actual_position_pct = actual_position_value / current_total_assets
        
        # Calculate final leverage after buy
        final_cash = self.cash - (shares * actual_buy_price + self.calculate_commission(shares))
        final_cash_ratio = final_cash / total_assets_after if total_assets_after > 0 else 0
        
        print(f"  [BUY] {symbol}: {shares} shares @ ${entry_price:.2f} (actual: ${actual_buy_price:.2f}) at {entry_time}")
        print(f"       Position: {actual_position_pct:.1%} of assets (${current_total_assets:,.0f}), Cost: ${total_cost:,.2f}")
        print(f"       Cash: ${self.cash:,.2f} → ${final_cash:,.2f} (ratio: {final_cash_ratio:.1%}), Leverage: {leverage_ratio:.2f}x")
        
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
                option_chain_id=row.get('option_chain_id', '')  # Get option_chain_id if exists
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
        print("BACKTEST RESULTS - V6 EVENT-DRIVEN")
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
            trade_df.to_csv('backtest_trades_v6_event_driven.csv', index=False)
            print(f"\nDetailed trade log saved to 'backtest_trades_v6_event_driven.csv'")


def main():
    import sys
    
    # Get holding days from command line
    holding_days = 6
    if len(sys.argv) > 1:
        try:
            holding_days = int(sys.argv[1])
            print(f"Using holding_days = {holding_days}")
        except ValueError:
            print(f"Invalid holding_days argument, using default of 1")
    
    print(f"\nInitializing V6 Event-Driven Backtest:")
    print(f"  Initial capital: $100,000")
    print(f"  Holding days: {holding_days}")
    print(f"  Max daily position: 80%")
    print(f"  Min cash ratio: 20%")
    print(f"  Take profit: 25%")
    print(f"  Stop loss: -10%\n")
    
    backtest = OptionFlowBacktestV6(initial_capital=100000, holding_days=holding_days)
    
    option_file = "option_data/merged_deduplicated_2023M3_2025M9_all.csv"
    
    backtest.run_backtest(option_file)
    backtest.calculate_metrics()


if __name__ == "__main__":
    main()
