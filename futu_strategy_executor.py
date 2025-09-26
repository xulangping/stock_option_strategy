#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Option Flow Strategy Executor using Futu API
Monitors real_time_option/ folder for new CSV files and executes trades
Based on option_flow_backtest_v5_late_only.py strategy rules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import warnings
from futu import *
import logging
from pathlib import Path
import threading

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futu_strategy_executor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FutuStrategyExecutor:
    def __init__(self, initial_capital: float = 100000):
        """Initialize the strategy executor with Futu API"""
        # Futu API configuration
        self.pwd_unlock = '153811'
        self.host = '127.0.0.1'
        self.port = 11111
        self.trd_env = TrdEnv.SIMULATE  # ALWAYS use simulation
        self.acc_id = 16428245  # US simulation account ID
        
        # Initialize contexts
        self.quote_ctx = None
        self.trd_ctx = None
        
        # Strategy parameters (from V5)
        self.initial_capital = initial_capital
        self.positions = {}  # {symbol: {'order_id': str, 'entry_price': float, 'shares': int, 'entry_time': datetime}}
        self.daily_trades = {}  # {date: count}
        self.daily_position_used = {}  # {date: percentage}
        self.stock_blacklist = {}  # {symbol: blacklist_until_date}
        
        # Strategy constants
        self.max_daily_trades = 4
        self.max_daily_position = 0.80
        self.min_cash_ratio = 0.20
        self.take_profit = 0.25
        self.stop_loss = -0.10
        self.blacklist_days = 5
        self.entry_delay_minutes = 5
        self.trade_start_time = (15, 30)  # 3:30 PM
        
        # File monitoring
        self.option_folder = "real_time_option"
        self.processed_files = set()
        self.state_file = "futu_executor_state.json"
        
        # Load previous state if exists
        self.load_state()
        
        # Connect to Futu
        self.connect()
    
    def connect(self):
        """Connect to Futu OpenD"""
        try:
            # Initialize quote context
            self.quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            logger.info("Connected to Futu quote context")
            
            # Initialize trade context for US market
            self.trd_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=self.host,
            port=self.port,
            security_firm=SecurityFirm.FUTUSECURITIES
        )
            
            # Unlock trade (required for placing orders)
            ret, data = self.trd_ctx.unlock_trade(self.pwd_unlock)
            if ret == RET_OK:
                logger.info("Trade unlocked successfully")
            else:
                logger.error(f"Failed to unlock trade: {data}")
                raise Exception(f"Failed to unlock trade: {data}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Futu: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Futu OpenD"""
        if self.quote_ctx:
            self.quote_ctx.close()
        if self.trd_ctx:
            self.trd_ctx.close()
        logger.info("Disconnected from Futu")
    
    def load_state(self):
        """Load previous execution state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_files = set(state.get('processed_files', []))
                    self.stock_blacklist = {
                        symbol: datetime.strptime(date_str, '%Y-%m-%d').date()
                        for symbol, date_str in state.get('stock_blacklist', {}).items()
                    }
                    logger.info(f"Loaded state: {len(self.processed_files)} processed files, {len(self.stock_blacklist)} blacklisted stocks")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def save_state(self):
        """Save current execution state"""
        try:
            state = {
                'processed_files': list(self.processed_files),
                'stock_blacklist': {
                    symbol: date.strftime('%Y-%m-%d')
                    for symbol, date in self.stock_blacklist.items()
                },
                'last_update': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def convert_option_time_to_market_time(self, date_str: str, time_str: str) -> datetime:
        """Convert UTC+8 to UTC-4 (ET) and handle date adjustment"""
        option_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        
        # Check if time is before 12:00:00
        hour = int(time_str.split(':')[0])
        if hour < 12:
            option_datetime = option_datetime + timedelta(days=1)
        
        # Convert from UTC+8 to UTC-4 (12 hour difference)
        market_datetime = option_datetime - timedelta(hours=12)
        
        return market_datetime
    
    def is_late_trade_time(self, signal_time: datetime) -> bool:
        """Check if signal is after 3:30 PM ET"""
        return (signal_time.hour > self.trade_start_time[0] or 
                (signal_time.hour == self.trade_start_time[0] and 
                 signal_time.minute >= self.trade_start_time[1]))
    
    def format_us_symbol(self, symbol: str) -> str:
        """Format symbol for US market (e.g., 'AAPL' -> 'US.AAPL')"""
        if not symbol.startswith('US.'):
            return f"US.{symbol}"
        return symbol
    
    def get_account_info(self) -> dict:
        """Get current account information"""
        ret, data = self.trd_ctx.accinfo_query(trd_env=self.trd_env, acc_id=self.acc_id)
        if ret == RET_OK:
            return data.iloc[0].to_dict() if not data.empty else {}
        else:
            logger.error(f"Failed to get account info: {data}")
            return {}
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions from Futu"""
        ret, data = self.trd_ctx.position_list_query(trd_env=self.trd_env, acc_id=self.acc_id)
        if ret == RET_OK:
            return data
        else:
            logger.error(f"Failed to get positions: {data}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol - returns 0 if no quote permission"""
        symbol = self.format_us_symbol(symbol)
        ret, data = self.quote_ctx.get_market_snapshot([symbol])
        if ret == RET_OK and not data.empty:
            return float(data.iloc[0]['last_price'])
        else:
            # Don't log error, just return 0 (no quote permission)
            return 0.0
    
    def place_buy_order(self, symbol: str, shares: int, signal_time: datetime) -> str:
        """Place a market buy order"""
        symbol = self.format_us_symbol(symbol)
        
        try:
            # Place market order with RTH session
            ret, data = self.trd_ctx.place_order(
                price=0,  # Market order doesn't need price
                qty=shares,
                code=symbol,
                trd_side=TrdSide.BUY,
                order_type=OrderType.MARKET,
                trd_env=self.trd_env,
                acc_id=self.acc_id,  # Specify account ID
                session=Session.RTH  # Regular trading hours only
            )
            
            if ret == RET_OK:
                order_id = data['order_id'][0]
                logger.info(f"[BUY ORDER] {symbol}: {shares} shares, Order ID: {order_id}")
                
                # Track position
                self.positions[symbol] = {
                    'order_id': order_id,
                    'entry_time': signal_time,
                    'shares': shares,
                    'entry_price': 0  # Will be updated when order fills
                }
                
                # Update daily trades count
                date = datetime.now().date()
                if date not in self.daily_trades:
                    self.daily_trades[date] = 0
                self.daily_trades[date] += 1
                
                # Add to blacklist
                blacklist_until = date + timedelta(days=self.blacklist_days)
                self.stock_blacklist[symbol] = blacklist_until
                
                return order_id
            else:
                logger.error(f"Failed to place buy order for {symbol}: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Exception placing buy order for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol: str, shares: int, reason: str) -> str:
        """Place a market sell order"""
        symbol = self.format_us_symbol(symbol)
        
        try:
            ret, data = self.trd_ctx.place_order(
                price=0,  # Market order
                qty=shares,
                code=symbol,
                trd_side=TrdSide.SELL,
                order_type=OrderType.MARKET,
                trd_env=self.trd_env,
                acc_id=self.acc_id,  # Specify account ID
                session=Session.RTH  # Regular trading hours only
            )
            
            if ret == RET_OK:
                order_id = data['order_id'][0]
                logger.info(f"[SELL ORDER - {reason}] {symbol}: {shares} shares, Order ID: {order_id}")
                
                # Remove from positions
                if symbol in self.positions:
                    del self.positions[symbol]
                
                return order_id
            else:
                logger.error(f"Failed to place sell order for {symbol}: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Exception placing sell order for {symbol}: {e}")
            return None
    
    def check_exit_conditions(self):
        """Check all positions for exit conditions"""
        current_positions = self.get_positions()
        
        for symbol, position_info in list(self.positions.items()):
            # Get current price (skip if no quote permission)
            current_price = self.get_current_price(symbol)
            if current_price <= 0:
                # No quote permission, use position data from Futu
                continue
            
            # Find position in Futu data
            symbol_formatted = self.format_us_symbol(symbol)
            position_data = current_positions[current_positions['code'] == symbol_formatted]
            
            if position_data.empty:
                logger.warning(f"Position {symbol} not found in Futu data")
                continue
            
            entry_price = float(position_data.iloc[0]['cost_price'])
            shares = int(position_data.iloc[0]['qty'])
            
            # Update entry price if not set
            if position_info['entry_price'] == 0:
                position_info['entry_price'] = entry_price
            
            # Calculate return
            returns = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if returns <= self.stop_loss:
                logger.info(f"[STOP LOSS TRIGGERED] {symbol}: Return = {returns:.2%}")
                self.place_sell_order(symbol, shares, "STOP_LOSS")
                continue
            
            # Check take profit
            if returns >= self.take_profit:
                logger.info(f"[TAKE PROFIT TRIGGERED] {symbol}: Return = {returns:.2%}")
                self.place_sell_order(symbol, shares, "TAKE_PROFIT")
                continue
            
            # Check next day 3PM exit (US Eastern Time)
            now = datetime.now()
            entry_time = position_info['entry_time']
            
            # If it's next trading day and after 3PM ET
            if now.date() > entry_time.date() and now.hour >= 15:
                logger.info(f"[NEXT DAY EXIT] {symbol}: Return = {returns:.2%}")
                self.place_sell_order(symbol, shares, "NEXT_DAY_EXIT")
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        date = datetime.now().date()
        
        # Check daily trade limit
        if date in self.daily_trades and self.daily_trades[date] >= self.max_daily_trades:
            logger.info(f"Daily trade limit reached: {self.daily_trades[date]}/{self.max_daily_trades}")
            return False
        
        # Check daily position limit
        if date in self.daily_position_used and self.daily_position_used[date] >= self.max_daily_position:
            logger.info(f"Daily position limit reached: {self.daily_position_used[date]:.1%}")
            return False
        
        # Check cash ratio
        account_info = self.get_account_info()
        if account_info:
            total_assets = float(account_info.get('total_assets', 0))
            cash = float(account_info.get('cash', 0))
            if total_assets > 0:
                cash_ratio = cash / total_assets
                if cash_ratio < self.min_cash_ratio:
                    logger.info(f"Insufficient cash ratio: {cash_ratio:.1%} < {self.min_cash_ratio:.1%}")
                    return False
        
        return True
    
    def process_option_signal(self, row: pd.Series):
        """Process a single option flow signal"""
        symbol = row['underlying_symbol']
        
        # Convert signal time to market time
        signal_time = self.convert_option_time_to_market_time(row['date'], row['time'])
        
        logger.info(f"\nProcessing signal: {symbol} at {signal_time} (ET)")
        
        # Check if it's late trade time (after 3:30 PM)
        if not self.is_late_trade_time(signal_time):
            logger.info(f"Skipping - signal before 3:30 PM ET")
            return
        
        # Check if signal is too close to market close
        if signal_time.hour == 15 and signal_time.minute >= (59 - self.entry_delay_minutes):
            logger.info(f"Skipping - signal too close to market close")
            return
        
        # Check blacklist
        today = datetime.now().date()
        if symbol in self.stock_blacklist and today <= self.stock_blacklist[symbol]:
            logger.info(f"Skipping - {symbol} in blacklist until {self.stock_blacklist[symbol]}")
            return
        
        # Check if already have position
        symbol_formatted = self.format_us_symbol(symbol)
        if symbol_formatted in self.positions:
            logger.info(f"Skipping - already have position in {symbol}")
            return
        
        # Check if can open new position
        if not self.can_open_position():
            return
        
        # Calculate position size based on option premium
        option_premium = row['premium']
        position_pct = min(option_premium / 800000, 0.40)  # Max 40% per position
        
        # Check and adjust for daily position limit
        date = today
        current_daily_position = self.daily_position_used.get(date, 0.0)
        
        if current_daily_position + position_pct > self.max_daily_position:
            position_pct = self.max_daily_position - current_daily_position
            logger.info(f"Adjusted position size to {position_pct:.1%} to stay within daily limit")
        
        # Get account info and calculate shares
        account_info = self.get_account_info()
        if not account_info:
            logger.error("Failed to get account info")
            return
        
        available_cash = float(account_info.get('cash', 0))
        position_value = available_cash * position_pct
        
        # Use underlying price from option data
        current_price = float(row.get('underlying_price', 0))
        if current_price <= 0:
            # If no price in data, use a default estimate
            logger.warning(f"No underlying_price in data for {symbol}, using estimate")
            # Default prices for common stocks
            default_prices = {
                'AAPL': 225.0,
                'GOOG': 170.0,
                'GOOGL': 170.0,
                'MSFT': 430.0,
                'TSLA': 250.0,
                'NVDA': 120.0,
            }
            current_price = default_prices.get(symbol, 100.0)
        
        logger.info(f"Using underlying price ${current_price:.2f} from option data for {symbol}")
        shares = int(position_value / current_price)
        if shares <= 0:
            logger.info(f"Position too small: {shares} shares")
            return
        
        # Calculate entry time (5 minutes after option signal)
        entry_time = signal_time + timedelta(minutes=self.entry_delay_minutes)
        now = datetime.now()
        
        # Convert entry_time to local time for comparison
        # Signal time is in ET (UTC-4), local time is UTC+8 (12 hour difference)
        entry_time_local = entry_time + timedelta(hours=12)
        
        wait_seconds = (entry_time_local - now).total_seconds()
        
        if wait_seconds > 0:
            logger.info(f"Signal was at {signal_time} ET")
            logger.info(f"Will enter at {entry_time} ET ({entry_time_local.strftime('%H:%M:%S')} local)")
            logger.info(f"Waiting {wait_seconds:.0f} seconds...")
            time.sleep(wait_seconds)
        else:
            logger.info(f"Signal was at {signal_time} ET, entry time has passed, placing order immediately")
        
        # Place buy order
        logger.info(f"Placing buy order: {symbol} - {shares} shares (Position: {position_pct:.1%})")
        order_id = self.place_buy_order(symbol, shares, signal_time)
        
        if order_id:
            # Update daily position tracking
            if date not in self.daily_position_used:
                self.daily_position_used[date] = 0
            self.daily_position_used[date] += position_pct
            
            logger.info(f"Buy order placed successfully. Order ID: {order_id}")
        else:
            logger.error(f"Failed to place buy order for {symbol}")
    
    def scan_for_new_files(self):
        """Scan the real_time_option folder for new CSV files"""
        if not os.path.exists(self.option_folder):
            logger.warning(f"Folder {self.option_folder} does not exist")
            return []
        
        new_files = []
        for filename in os.listdir(self.option_folder):
            if filename.endswith('.csv') and filename not in self.processed_files:
                filepath = os.path.join(self.option_folder, filename)
                new_files.append(filepath)
                self.processed_files.add(filename)
        
        return new_files
    
    def process_csv_file(self, filepath: str):
        """Process a CSV file with option flow signals"""
        try:
            logger.info(f"Processing file: {filepath}")
            df = pd.read_csv(filepath)
            
            # Process each signal
            for idx, row in df.iterrows():
                try:
                    self.process_option_signal(row)
                except Exception as e:
                    logger.error(f"Error processing signal {idx}: {e}")
                    continue
            
            # Save state after processing
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("Starting Futu Strategy Executor...")
        logger.info(f"Monitoring folder: {self.option_folder}")
        logger.info(f"Trading environment: {self.trd_env}")
        
        try:
            while True:
                # Check for new CSV files
                new_files = self.scan_for_new_files()
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new file(s)")
                    for filepath in new_files:
                        self.process_csv_file(filepath)
                
                # Check exit conditions for existing positions - DISABLED (no quote permission)
                # if self.positions:
                #     self.check_exit_conditions()
                
                # Reset daily counters at midnight
                now = datetime.now()
                if now.hour == 0 and now.minute < 3:
                    self.daily_trades.clear()
                    self.daily_position_used.clear()
                    logger.info("Reset daily counters")
                
                # Clean up old blacklist entries
                today = now.date()
                self.stock_blacklist = {
                    symbol: date for symbol, date in self.stock_blacklist.items()
                    if date >= today
                }
                
                # Wait 3 minutes before next scan
                time.sleep(180)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.save_state()
            self.disconnect()
            logger.info("Executor stopped")


def main():
    """Main entry point"""
    executor = FutuStrategyExecutor(initial_capital=100000)
    
    try:
        executor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        executor.disconnect()


if __name__ == "__main__":
    main()
