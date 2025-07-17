#!/usr/bin/env python3
"""
Daily Price Fetcher - Extract unusual underlying symbols from alerts and fetch their price data
"""

import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Set
import time
import glob

class DailyPriceFetcher:
    def __init__(self):
        # Create directories
        self.alert_data_dir = Path("alert_data")
        self.price_data_dir = Path("price_data")
        self.price_data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track processed symbols to avoid duplicate work
        self.processed_symbols = set()
        self.load_processed_symbols()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('daily_price_fetcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_processed_symbols(self):
        """Load already processed symbols to avoid duplicate work"""
        processed_file = self.price_data_dir / "processed_symbols.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    self.processed_symbols = set(data.get('symbols', []))
                self.logger.info(f"Loaded {len(self.processed_symbols)} processed symbols")
            except Exception as e:
                self.logger.error(f"Failed to load processed symbols: {e}")
    
    def save_processed_symbols(self):
        """Save processed symbols"""
        processed_file = self.price_data_dir / "processed_symbols.json"
        try:
            data = {
                'symbols': list(self.processed_symbols),
                'last_updated': datetime.now().isoformat()
            }
            with open(processed_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save processed symbols: {e}")
    
    def get_alert_files(self, days_back: int = 7) -> List[Path]:
        """Get alert JSON files from the last N days"""
        files = []
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            alert_file = self.alert_data_dir / f"alerts_{date_str}.json"
            if alert_file.exists():
                files.append(alert_file)
        
        self.logger.info(f"Found {len(files)} alert files from last {days_back} days")
        return files
    
    def extract_unusual_symbols(self, alert_files: List[Path]) -> Set[str]:
        """Extract unusual underlying symbols from alert files"""
        unusual_symbols = set()
        
        for file_path in alert_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
                
                for alert in alerts:
                    # Check if it's a flow alert (unusual activity)
                    if alert.get('noti_type') == 'flow_alerts':
                        meta = alert.get('meta', {})
                        underlying_symbol = meta.get('underlying_symbol')
                        
                        if underlying_symbol and self.is_unusual_activity(alert):
                            unusual_symbols.add(underlying_symbol)
                
                self.logger.info(f"Extracted {len(unusual_symbols)} symbols from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
        
        return unusual_symbols
    
    def is_unusual_activity(self, alert: dict) -> bool:
        """Determine if the alert represents unusual activity"""
        meta = alert.get('meta', {})
        
        # Check various indicators of unusual activity
        volume = meta.get('volume', 0)
        open_interest = meta.get('open_interest', 0)
        vol_oi_ratio = float(meta.get('vol_oi_ratio', 0))
        total_premium = float(meta.get('total_premium', 0))
        
        # Define thresholds for unusual activity
        unusual_criteria = [
            volume > 1000,  # High volume
            vol_oi_ratio > 2.0,  # High volume to open interest ratio
            total_premium > 100000,  # High premium value
            meta.get('rule_name') in [
                'LowHistoricVolumeFloor',
                'RepeatedHits',
                'RepeatedHitsAscendingFill',
                'RepeatedHitsDescendingFill',
                'FloorTradeLargeCap',
                'FloorTradeMidCap',
                'SweepsFollowedByFloor'
            ]
        ]
        
        # Return True if any criteria is met
        return any(unusual_criteria)
    
    def fetch_price_data(self, symbol: str, days_back: int = 30, max_retries: int = 3) -> pd.DataFrame:
        """Fetch 5-minute price data for a symbol with retry mechanism"""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Fetch 5-minute data
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="5m"
                )
                
                if data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
                
                # Reset index to make datetime a column
                data = data.reset_index()
                
                # Add symbol column
                data['Symbol'] = symbol
                
                # Reorder columns
                columns = ['Symbol', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                data = data[columns]
                
                self.logger.info(f"Fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                    wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                    self.logger.warning(f"Rate limited for {symbol}, attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return pd.DataFrame()
        
        self.logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        return pd.DataFrame()
    
    def save_price_data(self, symbol: str, data: pd.DataFrame):
        """Save price data to CSV file"""
        if data.empty:
            return
        
        csv_file = self.price_data_dir / f"{symbol}_5min.csv"
        
        try:
            # If file exists, append new data
            if csv_file.exists():
                existing_data = pd.read_csv(csv_file)
                existing_data['Datetime'] = pd.to_datetime(existing_data['Datetime'])
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                
                # Combine and remove duplicates
                combined_data = pd.concat([existing_data, data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['Datetime'], keep='last')
                combined_data = combined_data.sort_values('Datetime')
                
                data = combined_data
            
            # Save to CSV
            data.to_csv(csv_file, index=False)
            self.logger.info(f"Saved {len(data)} records to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data for {symbol}: {e}")
    
    def create_summary_report(self, processed_symbols: Set[str]):
        """Create a summary report of processed symbols"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols_processed': len(processed_symbols),
            'symbols': sorted(list(processed_symbols)),
            'data_files_created': []
        }
        
        # Check which files were created
        for symbol in processed_symbols:
            csv_file = self.price_data_dir / f"{symbol}_5min.csv"
            if csv_file.exists():
                file_info = {
                    'symbol': symbol,
                    'file': str(csv_file),
                    'size_mb': round(csv_file.stat().st_size / (1024 * 1024), 2),
                    'records': len(pd.read_csv(csv_file)) if csv_file.exists() else 0
                }
                report['data_files_created'].append(file_info)
        
        # Save report
        report_file = self.price_data_dir / "daily_summary.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Created summary report: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to create summary report: {e}")
    
    def run_daily_fetch(self, days_back: int = 7, price_days: int = 30):
        """Run the daily price fetching process"""
        self.logger.info("Starting daily price fetching process...")
        
        # 1. Get alert files
        alert_files = self.get_alert_files(days_back)
        if not alert_files:
            self.logger.warning("No alert files found")
            return
        
        # 2. Extract unusual symbols
        unusual_symbols = self.extract_unusual_symbols(alert_files)
        if not unusual_symbols:
            self.logger.info("No unusual symbols found")
            return
        
        # 3. Filter out already processed symbols (optional - remove if you want to update existing data)
        new_symbols = unusual_symbols - self.processed_symbols
        if not new_symbols:
            self.logger.info("All symbols already processed")
            # Uncomment the next line if you want to re-fetch all symbols daily
            # new_symbols = unusual_symbols
        
        self.logger.info(f"Processing {len(new_symbols)} new symbols: {sorted(new_symbols)}")
        
        # 4. Fetch and save price data for each symbol
        processed_count = 0
        for symbol in new_symbols:
            self.logger.info(f"Processing symbol: {symbol}")
            
            # Fetch price data
            price_data = self.fetch_price_data(symbol, price_days)
            
            if not price_data.empty:
                # Save price data
                self.save_price_data(symbol, price_data)
                self.processed_symbols.add(symbol)
                processed_count += 1
            
            # Add longer delay to avoid rate limiting
            time.sleep(5)
        
        # 5. Save processed symbols and create summary
        self.save_processed_symbols()
        self.create_summary_report(new_symbols)
        
        self.logger.info(f"Daily price fetching completed. Processed {processed_count} symbols.")
    
    def run_update_mode(self, price_days: int = 1):
        """Run in update mode - update existing symbols with recent data"""
        self.logger.info("Running in update mode...")
        
        if not self.processed_symbols:
            self.logger.warning("No processed symbols found. Run daily fetch first.")
            return
        
        for symbol in self.processed_symbols:
            self.logger.info(f"Updating data for symbol: {symbol}")
            
            # Fetch recent price data
            price_data = self.fetch_price_data(symbol, price_days)
            
            if not price_data.empty:
                self.save_price_data(symbol, price_data)
            
            # Add longer delay
            time.sleep(5)
        
        self.logger.info("Update mode completed.")

def main():
    """Main function"""
    fetcher = DailyPriceFetcher()
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--update":
            # Update mode - update existing symbols with recent data
            fetcher.run_update_mode()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python daily_price_fetcher.py           # Run daily fetch")
            print("  python daily_price_fetcher.py --update  # Update existing symbols")
            print("  python daily_price_fetcher.py --help    # Show this help")
            return
    else:
        # Default: run daily fetch
        fetcher.run_daily_fetch()

if __name__ == "__main__":
    main() 