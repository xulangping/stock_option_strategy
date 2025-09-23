#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Flow Momentum Strategy Backtest V3 - Using Local Data
Optimized version using pre-downloaded stock price data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class OptionFlowBacktestV3:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades_history = []
        self.daily_trades = {}
        self.daily_position_used = {}  # Track daily position usage
        self.stock_blacklist = {}  # {symbol: blacklist_until_date}
        self.stock_data_cache = {}  # Cache loaded stock data
        
        # Strategy parameters
        self.max_daily_trades = 5
        self.max_daily_position = 0.80  # Max 80% position per day
        self.min_cash_ratio = 0.20
        self.take_profit = 0.20
        self.stop_loss = -0.10
        self.blacklist_days = 5  # Don't buy same stock for 5 days
        
        # Data paths
        self.stock_data_dir = "stock_data"
        
    def convert_option_time_to_market_time(self, date_str: str, time_str: str) -> datetime:
        """Convert UTC+8 to UTC-4 and handle date adjustment"""
        option_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        
        # Check if time is before 12:00:00
        hour = int(time_str.split(':')[0])
        if hour < 12:
            option_datetime = option_datetime + timedelta(days=1)
        
        # Convert from UTC+8 to UTC-4
        market_datetime = option_datetime - timedelta(hours=12)
        
        return market_datetime
    
    def get_next_trading_day_open(self, current_time: datetime) -> datetime:
        """Get next trading day at market open (9:30 AM ET)"""
        next_day = current_time.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
            
        return next_day
    
    def load_stock_data_for_signal(self, symbol: str, signal_time: datetime) -> pd.DataFrame:
        """Load stock data from local parquet files that covers the signal time"""
        # Find all files for this symbol
        symbol_files = []
        for filename in os.listdir(self.stock_data_dir):
            if filename.startswith(f"{symbol}_") and filename.endswith(".parquet"):
                symbol_files.append(filename)
        
        if not symbol_files:
            print(f"  No data files found for {symbol}")
            return pd.DataFrame()
        
        # Find the file that contains our signal time
        signal_date = signal_time.date()
        next_day_open = self.get_next_trading_day_open(signal_time)
        
        for filename in symbol_files:
            # Extract date range from filename
            # Format: SYMBOL_YYYY-MM-DD_YYYY-MM-DD.parquet
            parts = filename.replace('.parquet', '').split('_')
            if len(parts) >= 3:
                try:
                    start_date = datetime.strptime(parts[1], "%Y-%m-%d").date()
                    end_date = datetime.strptime(parts[2], "%Y-%m-%d").date()
                    
                    # Check if signal date falls within this file's range
                    if start_date <= signal_date <= end_date:
                        # Load the data
                        filepath = os.path.join(self.stock_data_dir, filename)
                        if filepath in self.stock_data_cache:
                            df = self.stock_data_cache[filepath].copy()
                        else:
                            df = pd.read_parquet(filepath)
                            # Cache the data for future use
                            self.stock_data_cache[filepath] = df.copy()
                        
                        # Filter to only include data from signal time onwards
                        df = df[df['datetime'] >= signal_time]
                        
                        # Check if we need data from the next file for next day open
                        if df.empty or df['datetime'].max() < next_day_open:
                            # Try to find and load the next file
                            next_filename = f"{symbol}_{next_day_open.date()}_{next_day_open.date()}.parquet"
                            for fname in symbol_files:
                                if fname.startswith(f"{symbol}_{next_day_open.date()}"):
                                    next_filepath = os.path.join(self.stock_data_dir, fname)
                                    if next_filepath in self.stock_data_cache:
                                        next_df = self.stock_data_cache[next_filepath].copy()
                                    else:
                                        next_df = pd.read_parquet(next_filepath)
                                        self.stock_data_cache[next_filepath] = next_df.copy()
                                    
                                    # Combine the data
                                    df = pd.concat([df, next_df], ignore_index=True)
                                    break
                        
                        print(f"  Loaded {len(df)} bars for {symbol} from {filename}")
                        return df
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    continue
        
        print(f"  No suitable data file found for {symbol} on {signal_date}")
        return pd.DataFrame()
    
    def execute_trade(self, symbol: str, signal_time: datetime, action: str, price: float, shares: int = 0):
        """Execute and record trades"""
        trade = {
            'symbol': symbol,
            'time': signal_time,
            'action': action,
            'price': price,
            'shares': shares,
            'value': price * shares
        }
        
        if action == 'BUY':
            self.capital -= trade['value']
            self.positions[symbol] = {
                'entry_price': price,
                'entry_time': signal_time,
                'shares': shares
            }
            
            trade_date = signal_time.date()
            if trade_date not in self.daily_trades:
                self.daily_trades[trade_date] = 0
            self.daily_trades[trade_date] += 1
            
        elif action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'NEXT_DAY_EXIT']:
            self.capital += trade['value']
            
            if symbol in self.positions:
                entry_price = self.positions[symbol]['entry_price']
                trade['pnl'] = (price - entry_price) * shares
                trade['return'] = (price - entry_price) / entry_price
                del self.positions[symbol]
        
        self.trades_history.append(trade)
    
    def simulate_position(self, symbol: str, signal_time: datetime, df: pd.DataFrame, option_premium: float) -> bool:
        """Simulate a position from entry to exit"""
        if df.empty:
            print(f"  Not enough data to trade {symbol}")
            return False
        
        # Find the first bar at or after signal time + 1 minute
        entry_time_target = signal_time + timedelta(minutes=1)
        
        # Find the first bar that is at or after the target entry time AND during regular hours
        entry_idx = None
        for idx, row in df.iterrows():
            if row['datetime'] >= entry_time_target:
                # Check if it's during regular market hours (9:30 AM - 4:00 PM ET)
                hour = row['datetime'].hour
                minute = row['datetime'].minute
                if (hour == 9 and minute >= 30) or (10 <= hour < 16):
                    entry_idx = idx
                    break
        
        if entry_idx is None:
            print(f"  No data available after signal time for {symbol}")
            return False
        
        # Get entry price (close of first minute after signal)
        entry_bar = df.loc[entry_idx]
        entry_price = entry_bar['close']
        entry_time = entry_bar['datetime']
        
        # Check if we have next day 9:30 AM data before entering position
        next_day_930 = entry_time.replace(hour=9, minute=30, second=0) + timedelta(days=1)
        while next_day_930.weekday() >= 5:
            next_day_930 += timedelta(days=1)
            
        has_next_day_data = False
        for idx in range(len(df)):
            bar = df.iloc[idx]
            bar_time = bar['datetime']
            if (bar_time.date() == next_day_930.date() and 
                ((bar_time.hour == 9 and bar_time.minute >= 30) or 
                 (10 <= bar_time.hour < 16))):
                has_next_day_data = True
                break
        
        if not has_next_day_data:
            print(f"  Skipping {symbol} - no next day 9:30 AM data available")
            return False
        
        # Calculate position size based on option premium
        # Position size = option premium / 1000000 (e.g., 100k premium = 10% position)
        position_pct = min(option_premium / 1000000, 0.3)  # Cap at 30% per position
        
        # Check daily position limit
        date = signal_time.date()
        current_daily_position = self.daily_position_used.get(date, 0.0)
        if current_daily_position >= self.max_daily_position:
            print(f"  Daily position limit reached ({current_daily_position:.1%} >= {self.max_daily_position:.1%})")
            return False
        
        # Adjust position size if it would exceed daily limit
        if current_daily_position + position_pct > self.max_daily_position:
            position_pct = self.max_daily_position - current_daily_position
            print(f"  Adjusted position size to {position_pct:.1%} to stay within daily limit")
        
        # Calculate actual position value
        total_position_value = sum(pos['shares'] * pos['entry_price'] for pos in self.positions.values())
        available_cash = self.capital - total_position_value
        position_value = available_cash * position_pct
        
        if position_value <= 0:
            print(f"  Insufficient funds for {symbol}")
            return False
        
        shares = int(position_value / entry_price)
        if shares == 0:
            return False
        
        # Update daily position tracking
        if date not in self.daily_position_used:
            self.daily_position_used[date] = 0
        self.daily_position_used[date] += position_pct
        
        # Execute buy
        print(f"  [BUY] {symbol}: {shares} shares @ ${entry_price:.2f} at {entry_time} (Position: {position_pct:.1%})")
        self.execute_trade(symbol, entry_time, 'BUY', entry_price, shares)
        
        # Add to blacklist for 5 days
        blacklist_until = entry_time.date() + timedelta(days=self.blacklist_days)
        self.stock_blacklist[symbol] = blacklist_until
        
        # Check each subsequent bar for exit conditions
        for idx in range(entry_idx + 1, len(df)):
            bar = df.iloc[idx]
            current_price = bar['close']
            current_time = bar['datetime']
            returns = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if returns <= self.stop_loss:
                print(f"  [STOP_LOSS] {symbol}: {shares} shares @ ${current_price:.2f} at {current_time} (Return: {returns:.2%})")
                self.execute_trade(symbol, current_time, 'STOP_LOSS', current_price, shares)
                return True
            
            # Check take profit
            if returns >= self.take_profit:
                print(f"  [TAKE_PROFIT] {symbol}: {shares} shares @ ${current_price:.2f} at {current_time} (Return: {returns:.2%})")
                self.execute_trade(symbol, current_time, 'TAKE_PROFIT', current_price, shares)
                return True
            
            # Check if it's next trading day open
            if current_time.date() > entry_time.date() and current_time.hour == 9 and current_time.minute >= 30:
                print(f"  [NEXT_DAY_EXIT] {symbol}: {shares} shares @ ${current_price:.2f} at {current_time} (Return: {returns:.2%})")
                self.execute_trade(symbol, current_time, 'NEXT_DAY_EXIT', current_price, shares)
                return True
        
        # If we reach here, we need to find the next day 9:30 AM price
        next_day_930 = entry_time.replace(hour=9, minute=30, second=0) + timedelta(days=1)
        
        # Skip weekends
        while next_day_930.weekday() >= 5:
            next_day_930 += timedelta(days=1)
        
        # Find the price at or after 9:30 AM next day during regular market hours
        for idx in range(len(df)):
            bar = df.iloc[idx]
            bar_time = bar['datetime']
            
            # Check if it's the next trading day and during regular market hours
            if (bar_time.date() == next_day_930.date() and 
                ((bar_time.hour == 9 and bar_time.minute >= 30) or 
                 (10 <= bar_time.hour < 16))):
                print(f"  [NEXT_DAY_EXIT] {symbol}: {shares} shares @ ${bar['close']:.2f} at {bar['datetime']} (Return: {(bar['close'] - entry_price) / entry_price:.2%})")
                self.execute_trade(symbol, bar['datetime'], 'NEXT_DAY_EXIT', bar['close'], shares)
                return True
        
        # If no next day 9:30 AM data found, skip this trade
        print(f"  [WARNING] No next day 9:30 AM data found for {symbol}, position remains open")
        return False
    
    def run_backtest(self, option_flows_file: str):
        """Run the backtest on option flow data"""
        print("Loading option flow data...")
        df = pd.read_csv(option_flows_file)
        
        # Convert all times to market time first
        df['market_time'] = df.apply(
            lambda row: self.convert_option_time_to_market_time(row['date'], row['time']), 
            axis=1
        )
        
        # Sort by market time to process in chronological order
        df = df.sort_values('market_time').reset_index(drop=True)
        
        print(f"\nProcessing {len(df)} option flow signals in chronological order...\n")
        
        # Process each option flow signal in chronological order
        for idx, row in df.iterrows():
            print(f"\n{'='*60}")
            print(f"Signal #{idx + 1}: {row['underlying_symbol']}")
            
            # Use the pre-calculated market time
            signal_time = row['market_time']
            symbol = row['underlying_symbol']
            
            print(f"  Signal time (UTC-4): {signal_time}")
            
            # Skip if signal is too close to market close (after 3:55 PM)
            if signal_time.hour == 15 and signal_time.minute >= 58:
                print(f"  Skipping - signal too close to market close")
                continue
            
            # Skip if in blacklist
            if symbol in self.stock_blacklist and signal_time.date() <= self.stock_blacklist[symbol]:
                print(f"  Skipping - {symbol} in blacklist until {self.stock_blacklist[symbol]}")
                continue
            
            # Skip if already have position
            if symbol in self.positions:
                print(f"  Skipping - already have position in {symbol}")
                continue
            
            # Check if can open new position
            if not self.can_open_position(signal_time.date()):
                print(f"  Skipping - daily limit reached or insufficient cash")
                continue
            
            # Load stock data from local files
            stock_data = self.load_stock_data_for_signal(symbol, signal_time)
            
            if not stock_data.empty:
                # Get option premium from row
                option_premium = row['premium']
                print(f"  Option premium: ${option_premium:,.0f}")
                
                # Simulate the position
                self.simulate_position(symbol, signal_time, stock_data, option_premium)
            else:
                print(f"  No data available for {symbol}")
        
        print(f"\n{'='*60}")
        print("Backtest completed!")
    
    def can_open_position(self, date: datetime.date) -> bool:
        """Check if we can open new position"""
        # Check daily trade limit (only first 5 trades per day)
        if date in self.daily_trades and self.daily_trades[date] >= self.max_daily_trades:
            print(f"  Skipping - already made {self.daily_trades[date]} trades today (limit: {self.max_daily_trades})")
            return False
        
        total_position_value = sum(pos['shares'] * pos['entry_price'] for pos in self.positions.values())
        available_cash = self.capital - total_position_value
        
        if available_cash < self.initial_capital * self.min_cash_ratio:
            return False
        
        return True
    
    def calculate_metrics(self):
        """Calculate and display backtest metrics"""
        # Calculate total P&L
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades_history)
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Get winning and losing trades
        completed_trades = [t for t in self.trades_history if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        
        # Calculate metrics
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_return = np.mean([t['return'] for t in completed_trades]) if completed_trades else 0
        
        # Calculate max drawdown
        equity_curve = [self.initial_capital]
        current_equity = self.initial_capital
        
        for trade in self.trades_history:
            if trade['action'] == 'BUY':
                current_equity -= trade['value']
            else:
                current_equity += trade['value']
            equity_curve.append(current_equity)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"\nTotal Trades: {len(completed_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Average Return per Trade: {avg_return:.2%}")
        
        # Trade breakdown by exit type
        exit_types = {}
        for trade in self.trades_history:
            if trade['action'] in ['STOP_LOSS', 'TAKE_PROFIT', 'NEXT_DAY_EXIT', 'FORCED_EXIT']:
                exit_types[trade['action']] = exit_types.get(trade['action'], 0) + 1
        
        print("\nExit Type Breakdown:")
        for exit_type, count in exit_types.items():
            print(f"  {exit_type}: {count}")
        
        # Save detailed trade log
        if self.trades_history:
            trade_df = pd.DataFrame(self.trades_history)
            trade_df.to_csv('backtest_trades_v3_local.csv', index=False)
            print(f"\nDetailed trade log saved to 'backtest_trades_v3_local.csv'")
        
        # Print cache statistics
        print(f"\nCache Statistics:")
        print(f"  Files cached: {len(self.stock_data_cache)}")
        total_rows = sum(len(df) for df in self.stock_data_cache.values())
        print(f"  Total rows cached: {total_rows:,}")


def main():
    # Initialize backtest
    backtest = OptionFlowBacktestV3(initial_capital=100000)
    
    # Run backtest with the merged option data
    option_file = "option_data/merged_deduplicated_2025M1_M9.csv"
    
    backtest.run_backtest(option_file)
    
    # Calculate and display metrics
    backtest.calculate_metrics()


if __name__ == "__main__":
    main()
