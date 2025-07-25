import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import pytz
import re

class OptionFlowBacktester:
    def __init__(self, alert_data_dir: str, price_data_dir: str):
        self.alert_data_dir = alert_data_dir
        self.price_data_dir = price_data_dir
        self.alerts = []
        self.price_data = {}
        
    def load_alerts(self) -> List[Dict]:
        """Load alert data from all alert files with alerts_yyyy-mm-dd.json format"""
        alerts = []
        
        if os.path.exists(self.alert_data_dir):
            # Get all files matching the pattern alerts_yyyy-mm-dd.json
            for filename in os.listdir(self.alert_data_dir):
                if filename.startswith('alerts_') and filename.endswith('.json'):
                    # Verify the filename matches the expected date format
                    try:
                        date_part = filename.replace('alerts_', '').replace('.json', '')
                        datetime.strptime(date_part, '%Y-%m-%d')  # Validate date format
                        
                        filepath = os.path.join(self.alert_data_dir, filename)
                        with open(filepath, 'r') as f:
                            daily_alerts = json.load(f)
                            alerts.extend(daily_alerts)
                            print(f"Loaded {len(daily_alerts)} alerts from {filename}")
                    except ValueError:
                        continue  # Skip files that don't match the date format
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        print(f"Total loaded alerts: {len(alerts)}")
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
    
    def calculate_abnormality_score(self, alert: Dict, current_time: datetime) -> float:
        """Calculate abnormality score based on premium and days to expiration"""
        try:
            # Get trading premium
            premium = float(alert['meta'].get('total_premium', 0))
            
            # Parse option symbol to get expiration
            option_info = self.parse_option_symbol(alert['symbol'])
            exp_date = option_info['expiration']
            
            if exp_date is None:
                print(f"Failed to parse expiration for {alert['symbol']}")
                return 0
            
            # Calculate days to expiration (make exp_date timezone-aware)
            et_tz = pytz.timezone('US/Eastern')
            exp_date_aware = et_tz.localize(exp_date)
            days_to_exp = (exp_date_aware - current_time).days
            if days_to_exp <= 0:
                print(f"Expired option: {alert['symbol']}, days_to_exp: {days_to_exp}")
                return 0
            
            # Abnormality score: higher premium and shorter expiration = higher score
            # Use premium / days_to_exp as base score
            score = premium / max(days_to_exp, 1)
            
            return score
        except Exception as e:
            print(f"Error calculating score for {alert.get('symbol', 'unknown')}: {e}")
            return 0
    
    def select_top_abnormal_per_day(self, flow_alerts: List[Dict]) -> List[Dict]:
        """Select top 10 most abnormal alerts per trading day with deduplication"""
        # First, deduplicate alerts based on symbol, executed_time, and underlying_symbol
        seen_alerts = set()
        deduplicated_alerts = []
        
        for alert in flow_alerts:
            try:
                executed_time_str = alert['meta']['executed_at']
                underlying_symbol = alert['meta']['underlying_symbol']
                option_symbol = alert['symbol']
                
                # Create a unique key for deduplication
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
        
        daily_alerts = {}
        
        for alert in deduplicated_alerts:
            try:
                executed_time = self.convert_utc_to_et(alert['meta']['executed_at'])
                trade_date = executed_time.date()
                
                # Calculate abnormality score
                score = self.calculate_abnormality_score(alert, executed_time)
                
                if trade_date not in daily_alerts:
                    daily_alerts[trade_date] = []
                
                daily_alerts[trade_date].append({
                    'alert': alert,
                    'score': score,
                    'executed_time': executed_time
                })
            except Exception as e:
                print(f"Error processing alert {alert.get('id', 'unknown')}: {e}")
                continue
        
        # Select top 10 alerts for each day
        selected_alerts = []
        for date, alerts_list in daily_alerts.items():
            if alerts_list:
                # Sort by score (descending) and select top 10
                valid_alerts = [alert for alert in alerts_list if alert['score'] > 0]
                valid_alerts.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top 10 or all if less than 10
                top_alerts = valid_alerts[:10]
                
                print(f"Day {date}: {len(alerts_list)} total alerts, {len(valid_alerts)} valid, selected top {len(top_alerts)}")
                
                for i, alert_info in enumerate(top_alerts):
                    print(f"  #{i+1}: {alert_info['alert']['meta']['underlying_symbol']} "
                          f"(score: {alert_info['score']:.0f}, premium: ${float(alert_info['alert']['meta']['total_premium']):,.0f})")
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
    
    def is_after_hours(self, trade_time: datetime) -> bool:
        """Check if the trade time is after market hours (after 4:00 PM ET)"""
        # Market closes at 4:00 PM ET
        market_close_time = trade_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return trade_time.time() > market_close_time.time()
    
    def determine_trade_direction(self, alert: Dict) -> str:
        """
        Determine if the option trade is long or short based on price vs bid/ask and option type
        For Call options: price closer to ask = buy call (long), price closer to bid = sell call (short)
        For Put options: price closer to ask = buy put (short), price closer to bid = sell put (long)
        Returns: 'long' or 'short'
        """
        try:
            price = float(alert['meta'].get('price', 0))
            bid = float(alert['meta'].get('bid', 0))
            ask = float(alert['meta'].get('ask', 0))
            option_symbol = alert.get('symbol', '')
            
            if price == 0 or (bid == 0 and ask == 0):
                # Default to long if we can't determine
                return 'long'
            
            # Determine if it's a call or put option
            is_put = 'P' in option_symbol  # Put options have 'P' in symbol, calls have 'C'
            
            # Calculate mid-point and determine if closer to bid or ask
            if bid > 0 and ask > 0:
                closer_to_ask = abs(price - ask) < abs(price - bid)
                
                if is_put:
                    # For PUT options:
                    # Price closer to ask = buy put (short underlying)
                    # Price closer to bid = sell put (long underlying)
                    return 'short' if closer_to_ask else 'long'
                else:
                    # For CALL options:
                    # Price closer to ask = buy call (long underlying)
                    # Price closer to bid = sell call (short underlying)
                    return 'long' if closer_to_ask else 'short'
            
            # Fallback: if price < bid, it's likely a sell
            if bid > 0 and price < bid:
                return 'short' if not is_put else 'long'
            
            # Default to long
            return 'long'
            
        except (ValueError, TypeError):
            return 'long'
    
    def monitor_trade_new_exit_rule(self, symbol: str, entry_price: float, entry_time: datetime, 
                                   trade_date: datetime, trade_direction: str = 'long') -> Dict:
        """Monitor trade with new exit rule: sell next day open if available, otherwise current day close"""
        if symbol not in self.price_data:
            return None
            
        df = self.price_data[symbol]
        
        # First, try to get next day open price
        next_day = trade_date.date() + timedelta(days=1)
        et_tz = pytz.timezone('US/Eastern')
        
        # Look for next trading day prices (starting from 9:30 AM ET)
        next_day_start = et_tz.localize(datetime.combine(next_day, datetime.min.time().replace(hour=9, minute=30)))
        next_day_prices = df[df.index >= next_day_start]
        
        # Check if we have next day data
        if not next_day_prices.empty:
            # We have next day data, exit at next day open
            exit_price = next_day_prices.iloc[0]['Open']
            exit_time = next_day_prices.index[0]
            exit_reason = 'next_day_open'
            
            print(f"  -> Exiting at next day open: {exit_price:.2f} at {exit_time}")
            
            return {
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason
            }
        else:
            # No next day data, exit at current day close
            same_day_prices = df[(df.index >= entry_time) & (df.index.date == trade_date.date())]
            
            if same_day_prices.empty:
                return None
            
            # Exit at end of current day using Open price
            final_price = same_day_prices.iloc[-1]['Open']
            final_time = same_day_prices.index[-1]
            exit_reason = 'current_day_close'
            
            print(f"  -> No next day data, exiting at current day close: {final_price:.2f} at {final_time}")
            
            return {
                'exit_price': final_price,
                'exit_time': final_time,
                'exit_reason': exit_reason
            }
    
    def run_backtest(self) -> Dict:
        """Run the backtesting strategy"""
        print("Loading alerts...")
        self.alerts = self.load_alerts()
        
        print("Loading price data...")
        self.price_data = self.load_price_data()
        
        print("Filtering flow alerts...")
        flow_alerts = self.filter_flow_alerts(self.alerts)
        
        print(f"Found {len(flow_alerts)} 'Flow alerts for All' alerts")
        
        print("Selecting top 10 abnormal alerts per day...")
        selected_alerts = self.select_top_abnormal_per_day(flow_alerts)
        
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
                option_symbol = alert.get('symbol', '')
                option_type = 'PUT' if 'P' in option_symbol else 'CALL'
                print(f"Option details: {option_type} price=${option_price:.2f}, bid=${bid:.2f}, ask=${ask:.2f} -> {trade_direction.upper()}")
                
                # Check if this is an after-hours signal
                if self.is_after_hours(executed_time):
                    print(f"After-hours signal detected for {underlying_symbol}, using next day open price")
                    # Use next trading day open price
                    buy_price, actual_buy_time = self.get_next_trading_day_open(underlying_symbol, executed_time)
                    if buy_price is None:
                        print(f"Skipping {underlying_symbol} - no next day open price available")
                        continue
                    buy_time = actual_buy_time
                    trade_date = buy_time.date()
                else:
                    # Normal intraday signal
                    buy_price = self.get_price_at_time(underlying_symbol, buy_time)
                    if buy_price is None:
                        print(f"Skipping {underlying_symbol} - no buy price available")
                        continue
                
                # Monitor trade with new exit rule
                exit_info = self.monitor_trade_new_exit_rule(underlying_symbol, buy_price, buy_time, buy_time, trade_direction)
                if exit_info is None:
                    print(f"Skipping {underlying_symbol} - no exit price available")
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
                    'abnormality_score': alert_info['score'],
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
        
        # Calculate position sizes: 20% each, max 5 trades per day
        
        for date, day_trades in daily_trades.items():
            num_trades = len(day_trades)
            
            # If more than 5 trades, select only top 5 by abnormality score
            if num_trades > 5:
                day_trades.sort(key=lambda x: x['abnormality_score'], reverse=True)
                day_trades = day_trades[:5]
                print(f"Date {date}: Limited from {num_trades} to 5 trades (selected top 5 by score)")
                num_trades = 5
            
            position_size = 0.2  # Always 20% each (up to 5 trades per day)
            
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
        print(f"{'Date':<12} {'Symbol':<6} {'Type':<5} {'Dir':<5} {'Buy Time':<17} {'Buy Price':<10} {'Sell Price':<10} {'Return':<8} {'Exit Reason':<15} {'Score':<8}")
        print("-" * 150)
        
        for trade in results['trades']:
            direction = trade.get('trade_direction', 'long').upper()[:4]
            option_type = trade.get('option_type', 'CALL')[:4]  # Show first 4 chars (CALL/PUT)
            buy_time_str = trade['buy_time'].strftime('%m-%d %H:%M')  # Format as MM-DD HH:MM
            print(f"{trade['date']!s:<12} {trade['symbol']:<6} {option_type:<5} {direction:<5} {buy_time_str:<17} {trade['buy_price']:<10.2f} "
                  f"{trade['sell_price']:<10.2f} {trade['return_pct']:<8.2%} {trade['exit_reason']:<15} {trade['abnormality_score']:<8.0f}")

def main():
    # Initialize backtester
    backtester = OptionFlowBacktester('alert_data', 'price_data')
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print results
    backtester.print_results(results)
    
    # Save detailed results to JSON
    with open('backtest_results_v3.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        for trade in json_results['trades']:
            trade['date'] = trade['date'].isoformat()
            trade['alert_time'] = trade['alert_time'].isoformat()
            trade['buy_time'] = trade['buy_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to 'backtest_results_v3.json'")

if __name__ == "__main__":
    main() 