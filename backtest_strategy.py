import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import pytz
import re

class OptionFlowBacktester:
    def __init__(self, alert_data_dir: str, price_data_dir: str, max_trades_per_day: int = 4, min_premium: float = 500000, 
                 exit_strategy: str = 'next_day_open'):
        self.alert_data_dir = alert_data_dir
        self.price_data_dir = price_data_dir
        self.max_trades_per_day = max_trades_per_day
        self.min_premium = min_premium  # Minimum premium (500000 = $500,000)
        self.exit_strategy = exit_strategy  # 'next_day_open' or 'same_day_close'
        self.alerts = []
        self.price_data = {}
        
    def load_alerts(self) -> List[Dict]:
        """Load alert data from 2025-07-16 onwards, but only process those with bid/ask keys for direction determination"""
        alerts = []
        cutoff_date = datetime(2025, 7, 15)  # Load data from this date onwards
        
        if os.path.exists(self.alert_data_dir):
            # Get all files matching the pattern alerts_yyyy-mm-dd.json
            for filename in os.listdir(self.alert_data_dir):
                if filename.startswith('alerts_') and filename.endswith('.json'):
                    # Verify the filename matches the expected date format
                    try:
                        date_part = filename.replace('alerts_', '').replace('.json', '')
                        file_date = datetime.strptime(date_part, '%Y-%m-%d')  # Validate date format
                        
                        # Only load files from cutoff date onwards
                        if file_date >= cutoff_date:
                            filepath = os.path.join(self.alert_data_dir, filename)
                            with open(filepath, 'r') as f:
                                daily_alerts = json.load(f)
                                
                                # Filter alerts that have bid and ask keys for direction determination
                                valid_alerts = []
                                for alert in daily_alerts:
                                    meta = alert.get('meta', {})
                                    if ('bid' in meta and 'ask' in meta and 
                                        meta.get('bid') != '0' and meta.get('ask') != '0'):
                                        valid_alerts.append(alert)
                                
                                alerts.extend(valid_alerts)
                                print(f"Loaded {len(daily_alerts)} alerts from {filename}, {len(valid_alerts)} have bid/ask data")
                        else:
                            print(f"Skipped {filename} (before {cutoff_date.strftime('%Y-%m-%d')})")
                    except ValueError:
                        continue  # Skip files that don't match the date format
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        print(f"Total loaded alerts from {cutoff_date.strftime('%Y-%m-%d')} onwards with bid/ask data: {len(alerts)}")
        return alerts
    
    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data for all symbols"""
        price_data = {}
        for filename in os.listdir(self.price_data_dir):
            if filename.endswith('_5min.csv'):
                symbol = filename.replace('_5min.csv', '')
                filepath = os.path.join(self.price_data_dir, filename)
                df = pd.read_csv(filepath)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
                price_data[symbol] = df
        return price_data
    
    def filter_flow_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Filter alerts for 'Flow alerts for All' and 'Flow alerts for All put' rules"""
        filtered = []
        target_names = ['Flow alerts for All', 'Flow alerts for All put']
        
        for alert in alerts:
            alert_name = alert.get('name', '')
            if alert_name in target_names:
                filtered.append(alert)
        return filtered
    
    def parse_option_symbol(self, symbol: str) -> Dict:
        """Parse option symbol to extract expiration date"""
        # Example: UPST250718C00070000 -> expires 2025-07-18
        try:
            # Extract date part (YYMMDD format) - looking for 6 digits after symbol
            date_match = re.search(r'[A-Z]+(\d{6})[CP]', symbol)
            if date_match:
                date_str = date_match.group(1)
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                exp_date = datetime(year, month, day)
                return {'expiration': exp_date}
        except Exception as e:
            print(f"Error parsing option symbol {symbol}: {e}")
        return {'expiration': None}
    

    
    def select_first_n_per_day(self, flow_alerts: List[Dict]) -> List[Dict]:
        """Select first N alerts per trading day by time order to avoid lookahead bias"""
        # First, deduplicate alerts based on symbol, executed_time, and underlying_symbol
        seen_alerts = set()
        deduplicated_alerts = []
        
        for alert in flow_alerts:
            try:
                executed_time_str = alert['meta']['executed_at']
                underlying_symbol = alert['meta']['underlying_symbol']
                option_symbol = alert['symbol']
                
                # Create a unique key for deduplication (ignore name, only use symbol)
                alert_key = (executed_time_str, underlying_symbol, option_symbol)
                
                if alert_key not in seen_alerts:
                    seen_alerts.add(alert_key)
                    deduplicated_alerts.append(alert)
                else:
                    print(f"Skipping duplicate alert: {underlying_symbol} at {executed_time_str}")
            except Exception as e:
                print(f"Error processing alert for deduplication: {e}")
                continue
        
        print(f"Deduplicated {len(flow_alerts)} alerts to {len(deduplicated_alerts)} unique alerts")
        
        # Group alerts by trading day
        daily_alerts = {}
        
        for alert in deduplicated_alerts:
            try:
                executed_time = self.convert_utc_to_et(alert['meta']['executed_at'])
                trade_date = executed_time.date()
                
                if trade_date not in daily_alerts:
                    daily_alerts[trade_date] = []
                
                daily_alerts[trade_date].append({
                    'alert': alert,
                    'executed_time': executed_time
                })
            except Exception as e:
                print(f"Error processing alert {alert.get('id', 'unknown')}: {e}")
                continue
        
        # Select first 4 alerts for each day by time order (not by score)
        selected_alerts = []
        for date, alerts_list in daily_alerts.items():
            if alerts_list:
                                # Filter by premium and data availability, then sort by execution time (ascending - earliest first)
                valid_alerts = []
                for alert in alerts_list:
                    underlying_symbol = alert['alert']['meta']['underlying_symbol']
                    premium = float(alert['alert']['meta'].get('total_premium', 0))
                    
                    # Check if premium meets minimum requirement
                    if premium < self.min_premium:
                        print(f"    Filtered out {underlying_symbol} - premium ${premium:.0f} < ${self.min_premium:.0f}")
                        continue
                    
                    # Always ensure we have next day data for consistency across strategies
                    if not self.has_next_day_data(underlying_symbol, alert['executed_time']):
                        print(f"    Filtered out {underlying_symbol} - no next day data (for consistency)")
                        continue
                    
                    valid_alerts.append(alert)
                
                valid_alerts.sort(key=lambda x: x['executed_time'])
                
                # Take first N by time order (not by score)
                first_n_alerts = valid_alerts[:self.max_trades_per_day]
                
                print(f"Day {date}: {len(alerts_list)} total alerts, {len(valid_alerts)} valid (premium >= ${self.min_premium:.0f}, next day data available), selected first {len(first_n_alerts)} by time")
                
                for i, alert_info in enumerate(first_n_alerts):
                    premium = float(alert_info['alert']['meta']['total_premium'])
                    print(f"  #{i+1}: {alert_info['executed_time'].strftime('%H:%M')} - {alert_info['alert']['meta']['underlying_symbol']} "
                          f"(premium: ${premium:,.0f})")
                    selected_alerts.append(alert_info)
        
        return selected_alerts
    
    def convert_utc_to_et(self, utc_time_str: str) -> datetime:
        """Convert UTC time to Eastern Time"""
        utc_time = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
        utc_tz = pytz.UTC
        et_tz = pytz.timezone('US/Eastern')
        utc_time = utc_tz.localize(utc_time.replace(tzinfo=None))
        et_time = utc_time.astimezone(et_tz)
        return et_time
    
    def get_price_at_time(self, symbol: str, target_time: datetime) -> float:
        """Get price at specific time"""
        if symbol not in self.price_data:
            return None
            
        df = self.price_data[symbol]
        
        # Find the closest timestamp after target_time
        future_prices = df[df.index >= target_time]
        if future_prices.empty:
            # If no future prices, try to get the last available price on the same day
            same_day_prices = df[df.index.date == target_time.date()]
            if same_day_prices.empty:
                return None
            # Use the last available Open price of the day
            return same_day_prices.iloc[-1]['Open']
            
        return future_prices.iloc[0]['Open']
    
    def get_next_trading_day_open(self, symbol: str, after_time: datetime) -> tuple:
        """Get the opening price of the next trading day after given time"""
        if symbol not in self.price_data:
            return None, None
            
        df = self.price_data[symbol]
        
        # Find the next trading day (after 9:30 AM ET)
        next_day = after_time.date() + timedelta(days=1)
        
        # Look for prices on the next day starting from 9:30 AM
        et_tz = pytz.timezone('US/Eastern')
        next_day_start = et_tz.localize(datetime.combine(next_day, datetime.min.time().replace(hour=9, minute=30)))
        
        # Find the first price after market open on next day
        next_day_prices = df[df.index >= next_day_start]
        if next_day_prices.empty:
            return None, None
            
        open_price = next_day_prices.iloc[0]['Open']  # Use Open price instead of Close
        open_time = next_day_prices.index[0]
        
        return open_price, open_time
    
    def has_next_day_data(self, symbol: str, trade_date: datetime) -> bool:
        """Check if symbol has next day trading data"""
        next_day_open, _ = self.get_next_trading_day_open(symbol, trade_date)
        return next_day_open is not None
    
    def get_same_day_close(self, symbol: str, entry_time: datetime, trade_date: datetime) -> tuple:
        """Get the closing price of the same trading day"""
        if symbol not in self.price_data:
            return None, None
            
        df = self.price_data[symbol]
        
        # Get same day prices after entry time
        same_day_prices = df[(df.index >= entry_time) & (df.index.date == trade_date.date())]
        
        if same_day_prices.empty:
            return None, None
        
        # Get the last price of the day
        close_price = same_day_prices.iloc[-1]['Open']  # Use Open price
        close_time = same_day_prices.index[-1]
        
        return close_price, close_time
    
    def is_after_hours(self, trade_time: datetime) -> bool:
        """Check if the trade time is after market hours (after 4:00 PM ET)"""
        # Market closes at 4:00 PM ET
        market_close_time = trade_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return trade_time.time() > market_close_time.time()
    
    def determine_trade_direction(self, alert: Dict) -> str:
        """
        Determine if the option trade is long or short based on bid/ask volume or price
        For Call options: buy call = long, sell call = short
        For Put options: buy put = short, sell put = long
        Returns: 'long' or 'short'
        """
        try:
            option_symbol = alert.get('symbol', '')
            # More accurate option type detection
            if re.search(r'\d{6}P\d', option_symbol):  # 6 digits + P + digits
                is_put = True
            elif re.search(r'\d{6}C\d', option_symbol):  # 6 digits + C + digits
                is_put = False
            else:
                # Fallback to simple check
                is_put = 'P' in option_symbol
            
            # First try to use bid_volume and ask_volume (newer data format)
            bid_volume = alert['meta'].get('bid_volume', 0)
            ask_volume = alert['meta'].get('ask_volume', 0)
            
            if bid_volume is not None and ask_volume is not None and (bid_volume > 0 or ask_volume > 0):
                # Use volume to determine trade direction (more reliable)
                is_bid_trade = bid_volume > ask_volume  # More volume on bid = seller behavior
                trade_type = "SELL" if is_bid_trade else "BUY"
                
                if is_put:
                    # PUT: sell put (bid) = long, buy put (ask) = short
                    direction = 'long' if is_bid_trade else 'short'
                    print(f"    Volume-based: {trade_type} PUT (bid_vol={bid_volume}, ask_vol={ask_volume}) -> {direction.upper()}")
                    return direction
                else:
                    # CALL: buy call (ask) = long, sell call (bid) = short
                    direction = 'short' if is_bid_trade else 'long'
                    print(f"    Volume-based: {trade_type} CALL (bid_vol={bid_volume}, ask_vol={ask_volume}) -> {direction.upper()}")
                    return direction
            
            # Fallback to price vs bid/ask method (older data format)
            price = float(alert['meta'].get('price', 0))
            bid = float(alert['meta'].get('bid', 0))
            ask = float(alert['meta'].get('ask', 0))
            
            if price == 0 or (bid == 0 and ask == 0):
                return 'long'  # Default to long if we can't determine
            
            # Calculate mid-point and determine if closer to bid or ask
            if bid > 0 and ask > 0:
                closer_to_ask = abs(price - ask) < abs(price - bid)
                trade_type = "BUY" if closer_to_ask else "SELL"
                
                if is_put:
                    # PUT: price closer to ask = buy put (short), price closer to bid = sell put (long)
                    direction = 'short' if closer_to_ask else 'long'
                    print(f"    Price-based: {trade_type} PUT (price=${price:.2f}, bid=${bid:.2f}, ask=${ask:.2f}) -> {direction.upper()}")
                    return direction
                else:
                    # CALL: price closer to ask = buy call (long), price closer to bid = sell call (short)
                    direction = 'long' if closer_to_ask else 'short'
                    print(f"    Price-based: {trade_type} CALL (price=${price:.2f}, bid=${bid:.2f}, ask=${ask:.2f}) -> {direction.upper()}")
                    return direction
            
            # Final fallback
            return 'long'
            
        except (ValueError, TypeError) as e:
            print(f"Error determining trade direction: {e}")
            return 'long'
    
    def monitor_trade_exit(self, symbol: str, entry_price: float, entry_time: datetime, 
                          trade_date: datetime, trade_direction: str = 'long') -> Dict:
        """Monitor trade with configurable exit strategy"""
        if symbol not in self.price_data:
            return None
            
        if self.exit_strategy == 'next_day_open':
            # Next day open strategy
            next_day_open, next_day_time = self.get_next_trading_day_open(symbol, trade_date)
            if next_day_open is not None:
                print(f"  -> Exiting at next day open: {next_day_open:.2f} at {next_day_time}")
                return {
                    'exit_price': next_day_open,
                    'exit_time': next_day_time,
                    'exit_reason': 'next_day_open'
                }
            else:
                print(f"  -> No next day data available for {symbol}")
                return None
        
        elif self.exit_strategy == 'same_day_close':
            # Same day close strategy
            close_price, close_time = self.get_same_day_close(symbol, entry_time, trade_date)
            if close_price is not None:
                print(f"  -> Exiting at same day close: {close_price:.2f} at {close_time}")
                return {
                    'exit_price': close_price,
                    'exit_time': close_time,
                    'exit_reason': 'same_day_close'
                }
            else:
                print(f"  -> No same day close data available for {symbol}")
                return None
        
        else:
            raise ValueError(f"Unknown exit strategy: {self.exit_strategy}")
    
    # Keep the old function for backward compatibility
    def monitor_trade_new_exit_rule(self, symbol: str, entry_price: float, entry_time: datetime, 
                                   trade_date: datetime, trade_direction: str = 'long') -> Dict:
        """Legacy function - use monitor_trade_exit instead"""
        return self.monitor_trade_exit(symbol, entry_price, entry_time, trade_date, trade_direction)
    
    def run_backtest(self) -> Dict:
        """Run the backtesting strategy"""
        print("Loading alerts...")
        self.alerts = self.load_alerts()
        
        print("Loading price data...")
        self.price_data = self.load_price_data()
        
        print("Filtering flow alerts...")
        flow_alerts = self.filter_flow_alerts(self.alerts)
        
        print(f"Found {len(flow_alerts)} 'Flow alerts for All' alerts")
        
        print(f"Selecting first {self.max_trades_per_day} alerts per day by time order (minimum premium: ${self.min_premium:.0f}, exit strategy: {self.exit_strategy})...")
        selected_alerts = self.select_first_n_per_day(flow_alerts)
        
        print(f"Selected {len(selected_alerts)} alerts for backtesting")
        
        trades = []
        daily_trades = {}
        
        for alert_info in selected_alerts:
            try:
                alert = alert_info['alert']
                executed_time = alert_info['executed_time']
                buy_time = executed_time + timedelta(minutes=5)
                trade_date = buy_time.date()
                
                # Extract underlying symbol
                underlying_symbol = alert['meta']['underlying_symbol']
                
                print(f"Processing alert for {underlying_symbol} at {executed_time}")
                
                # Determine trade direction (long or short)
                trade_direction = self.determine_trade_direction(alert)
                option_price = float(alert['meta'].get('price', 0))
                bid = float(alert['meta'].get('bid', 0))
                ask = float(alert['meta'].get('ask', 0))
                bid_volume = alert['meta'].get('bid_volume', 'N/A')
                ask_volume = alert['meta'].get('ask_volume', 'N/A')
                option_symbol = alert.get('symbol', '')
                # Determine option type from symbol (more accurate parsing)
                # Format: SYMBOL + YYMMDD + C/P + STRIKE
                # Example: MCHP250808C00069000
                if re.search(r'\d{6}P\d', option_symbol):  # 6 digits + P + digits
                    option_type = 'PUT'
                elif re.search(r'\d{6}C\d', option_symbol):  # 6 digits + C + digits  
                    option_type = 'CALL'
                else:
                    # Fallback to simple check
                    option_type = 'PUT' if 'P' in option_symbol else 'CALL'
                
                # Show volume info if available
                if bid_volume != 'N/A' and ask_volume != 'N/A':
                    volume_info = f", bid_vol={bid_volume}, ask_vol={ask_volume}"
                else:
                    volume_info = f", bid=${bid:.2f}, ask=${ask:.2f}"
                
                print(f"Option details: {option_type} price=${option_price:.2f}{volume_info} -> {trade_direction.upper()}")
                
                # Check if this is an after-hours signal
                if self.is_after_hours(executed_time):
                    print(f"After-hours signal detected for {underlying_symbol}, using next day open price")
                    # Use next trading day open price
                    buy_price, actual_buy_time = self.get_next_trading_day_open(underlying_symbol, executed_time)
                    if buy_price is None:
                        print(f"❌ CRITICAL ERROR: {underlying_symbol} - no next day open price available for after-hours trade")
                        continue
                    buy_time = actual_buy_time
                    trade_date = buy_time.date()
                    print(f"✅ After-hours trade: {underlying_symbol} buy at {buy_price:.2f} on {buy_time}")
                else:
                    # Normal intraday signal
                    buy_price = self.get_price_at_time(underlying_symbol, buy_time)
                    if buy_price is None:
                        print(f"❌ CRITICAL ERROR: {underlying_symbol} - no buy price available for intraday trade at {buy_time}")
                        continue
                    print(f"✅ Intraday trade: {underlying_symbol} buy at {buy_price:.2f} at {buy_time}")
                
                # Monitor trade with configured exit strategy
                exit_info = self.monitor_trade_exit(underlying_symbol, buy_price, buy_time, buy_time, trade_direction)
                if exit_info is None:
                    print(f"❌ CRITICAL ERROR: {underlying_symbol} - no exit price available ({self.exit_strategy})")
                    print(f"   Trade date: {buy_time.date()}, Entry time: {buy_time}")
                    continue
                
                sell_price = exit_info['exit_price']
                exit_reason = exit_info['exit_reason']
                
                # Calculate return based on trade direction
                if trade_direction == 'long':
                    return_pct = (sell_price - buy_price) / buy_price
                else:
                    # For short trades, profit when price goes down
                    return_pct = (buy_price - sell_price) / buy_price
                
                # Track trades by date for position sizing
                if trade_date not in daily_trades:
                    daily_trades[trade_date] = []
                
                trade_info = {
                    'date': trade_date,
                    'symbol': underlying_symbol,
                    'alert_time': executed_time,
                    'buy_time': buy_time,
                    'exit_time': exit_info['exit_time'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return_pct': return_pct,
                    'exit_reason': exit_reason,
                    'alert_id': alert['id'],
                    'trade_direction': trade_direction,
                    'option_type': option_type,
                    'option_symbol': option_symbol,
                    'option_price': option_price,
                    'option_bid': bid,
                    'option_ask': ask
                }
                
                daily_trades[trade_date].append(trade_info)
                print(f"Added trade: {underlying_symbol} {return_pct:.2%} ({trade_direction.upper()}, exit: {exit_reason})")
                
            except Exception as e:
                print(f"Error processing alert {alert.get('id', 'unknown')}: {e}")
                continue
        
        # Calculate position sizes based on max_trades_per_day
        
        for date, day_trades in daily_trades.items():
            num_trades = len(day_trades)
            
            # Position size is 1 / max_trades_per_day (e.g., 25% for 4 trades, 20% for 5 trades)
            position_size = 1.0 / self.max_trades_per_day
            max_trades = self.max_trades_per_day
            
            if num_trades > max_trades:
                print(f"Warning: Date {date} has {num_trades} trades, but should only have {max_trades}")
                # This shouldn't happen since we select first N by time
                day_trades = day_trades[:max_trades]
                num_trades = max_trades
            
            print(f"Date {date}: {num_trades} trades, {position_size:.1%} each")
            
            for trade in day_trades:
                trade['position_size'] = position_size
                trade['weighted_return'] = trade['return_pct'] * position_size
                trades.append(trade)
        
        # Calculate overall performance
        if trades:
            total_return = sum(trade['weighted_return'] for trade in trades)
            num_trades = len(trades)
            win_rate = sum(1 for trade in trades if trade['return_pct'] > 0) / num_trades
            
            avg_return = np.mean([trade['return_pct'] for trade in trades])
            max_return = max([trade['return_pct'] for trade in trades])
            min_return = min([trade['return_pct'] for trade in trades])
            
            # Exit reason statistics
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        else:
            total_return = 0
            num_trades = 0
            win_rate = 0
            avg_return = 0
            max_return = 0
            min_return = 0
            exit_reasons = {}
        
        results = {
            'total_trades': num_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'max_return': max_return,
            'min_return': min_return,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Average Return per Trade: {results['avg_return_per_trade']:.2%}")
        print(f"Best Trade: {results['max_return']:.2%}")
        print(f"Worst Trade: {results['min_return']:.2%}")
        
        # Exit reason statistics
        print(f"\nExit Reasons:")
        for reason, count in results['exit_reasons'].items():
            print(f"  {reason}: {count} trades")
        
        # Show all trades
        print("\nAll Trades:")
        print("-" * 150)
        print(f"{'Date':<12} {'Symbol':<6} {'Type':<5} {'Dir':<5} {'Buy Time':<17} {'Buy Price':<10} {'Sell Price':<10} {'Return':<8} {'Exit Reason':<15}")
        print("-" * 140)
        
        for trade in results['trades']:
            direction = trade.get('trade_direction', 'long').upper()[:4]
            option_type = trade.get('option_type', 'CALL')[:4]  # Show first 4 chars (CALL/PUT)
            buy_time_str = trade['buy_time'].strftime('%m-%d %H:%M')  # Format as MM-DD HH:MM
            print(f"{trade['date']!s:<12} {trade['symbol']:<6} {option_type:<5} {direction:<5} {buy_time_str:<17} {trade['buy_price']:<10.2f} "
                  f"{trade['sell_price']:<10.2f} {trade['return_pct']:<8.2%} {trade['exit_reason']:<15}")

def main():
    # Initialize backtester with configurable parameters
    # You can change max_trades_per_day to test different strategies:
    # - 4 trades per day = 25% position size each
    # - 5 trades per day = 20% position size each  
    # - 3 trades per day = 33.3% position size each
    max_trades_per_day = 4  # Change this value to test different strategies
    
    # Filter by minimum premium: 500000 = $500,000
    min_premium = 500000  # Only select alerts with premium >= $500,000
    
    # Exit strategy: 'next_day_open' or 'same_day_close'
    # Both strategies use the same stock universe (only stocks with both current and next day data)
    # next_day_open: Exit at next trading day's open
    # same_day_close: Exit at same day's close
    exit_strategy = 'next_day_open'  # Change this to test different exit strategies
    
    backtester = OptionFlowBacktester('alert_data', 'price_data', 
                                     max_trades_per_day=max_trades_per_day,
                                     min_premium=min_premium,
                                     exit_strategy=exit_strategy)
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print results
    backtester.print_results(results)
    
    # Save detailed results to JSON
    filename = f'backtest_results_v4_time_order_max{max_trades_per_day}_min{int(min_premium)}_{exit_strategy}.json'
    with open(filename, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        for trade in json_results['trades']:
            trade['date'] = trade['date'].isoformat()
            trade['alert_time'] = trade['alert_time'].isoformat()
            trade['buy_time'] = trade['buy_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to '{filename}'")

if __name__ == "__main__":
    main() 