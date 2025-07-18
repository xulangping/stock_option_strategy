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
        """Load alert data from specific JSON files"""
        alerts = []
        target_files = ['alerts_2025-07-15.json', 'alerts_2025-07-16.json', 'alerts_2025-07-18.json']
        
        for filename in target_files:
            filepath = os.path.join(self.alert_data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    daily_alerts = json.load(f)
                    alerts.extend(daily_alerts)
                    print(f"Loaded {len(daily_alerts)} alerts from {filename}")
            else:
                print(f"Warning: {filename} not found")
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
        """Filter alerts for exactly 'Flow alerts for All' rule"""
        filtered = []
        for alert in alerts:
            alert_name = alert.get('name', '')
            # Only match exactly "Flow alerts for All"
            if alert_name == 'Flow alerts for All':
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
        """Select top 10 most abnormal alerts per trading day"""
        daily_alerts = {}
        
        for alert in flow_alerts:
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
            # Use the last available price of the day
            return same_day_prices.iloc[-1]['Close']
            
        return future_prices.iloc[0]['Close']
    
    def monitor_trade_with_stops(self, symbol: str, entry_price: float, entry_time: datetime, 
                                trade_date: datetime) -> Dict:
        """Monitor trade with stop-loss and take-profit"""
        if symbol not in self.price_data:
            return None
            
        df = self.price_data[symbol]
        
        # Get all prices from entry time to end of day
        same_day_prices = df[(df.index >= entry_time) & (df.index.date == trade_date.date())]
        
        if same_day_prices.empty:
            return None
        
        stop_loss_price = entry_price * 0.98  # -2% stop loss
        take_profit_price = entry_price * 1.05  # +5% take profit
        
        for timestamp, row in same_day_prices.iterrows():
            current_price = row['Close']
            
            # Check stop loss
            if current_price <= stop_loss_price:
                return {
                    'exit_price': current_price,
                    'exit_time': timestamp,
                    'exit_reason': 'stop_loss'
                }
            
            # Check take profit
            if current_price >= take_profit_price:
                return {
                    'exit_price': current_price,
                    'exit_time': timestamp,
                    'exit_reason': 'take_profit'
                }
        
        # If no stop triggered, exit at end of day
        final_price = same_day_prices.iloc[-1]['Close']
        final_time = same_day_prices.index[-1]
        
        return {
            'exit_price': final_price,
            'exit_time': final_time,
            'exit_reason': 'end_of_day'
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
                
                # Get buy price
                buy_price = self.get_price_at_time(underlying_symbol, buy_time)
                if buy_price is None:
                    print(f"Skipping {underlying_symbol} - no buy price available")
                    continue
                
                # Monitor trade with stop-loss and take-profit
                exit_info = self.monitor_trade_with_stops(underlying_symbol, buy_price, buy_time, buy_time)
                if exit_info is None:
                    print(f"Skipping {underlying_symbol} - no exit price available")
                    continue
                
                sell_price = exit_info['exit_price']
                exit_reason = exit_info['exit_reason']
                
                # Calculate return
                return_pct = (sell_price - buy_price) / buy_price
                
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
                    'abnormality_score': alert_info['score']
                }
                
                daily_trades[trade_date].append(trade_info)
                print(f"Added trade: {underlying_symbol} {return_pct:.2%} (exit: {exit_reason})")
                
            except Exception as e:
                print(f"Error processing alert {alert.get('id', 'unknown')}: {e}")
                continue
        
        # Calculate position sizes: 10% each, max 100% total
        for date, day_trades in daily_trades.items():
            num_trades = len(day_trades)
            if num_trades <= 10:
                position_size = 0.10  # 10% each
            else:
                position_size = 1.0 / num_trades  # Split 100% evenly
            
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
        print("-" * 100)
        print(f"{'Date':<12} {'Symbol':<6} {'Buy Price':<10} {'Sell Price':<10} {'Return':<8} {'Exit Reason':<12} {'Score':<8}")
        print("-" * 100)
        
        for trade in results['trades']:
            print(f"{trade['date']!s:<12} {trade['symbol']:<6} {trade['buy_price']:<10.2f} "
                  f"{trade['sell_price']:<10.2f} {trade['return_pct']:<8.2%} {trade['exit_reason']:<12} {trade['abnormality_score']:<8.0f}")

def main():
    # Initialize backtester
    backtester = OptionFlowBacktester('alert_data', 'price_data')
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print results
    backtester.print_results(results)
    
    # Save detailed results to JSON
    with open('backtest_results_v2.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        for trade in json_results['trades']:
            trade['date'] = trade['date'].isoformat()
            trade['alert_time'] = trade['alert_time'].isoformat()
            trade['buy_time'] = trade['buy_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to 'backtest_results_v2.json'")

if __name__ == "__main__":
    main() 