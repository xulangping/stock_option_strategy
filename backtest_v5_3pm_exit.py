#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Flow Backtest V5 - Trade after 3:30 PM, exit next day 3:00 PM
Simplified version for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Read previous results to compare
try:
    old_trades = pd.read_csv('backtest_trades_v5_late_only.csv')
    print(f"Previous results: {len(old_trades)} trades")
    print(f"Previous return: 13.35%")
except:
    print("No previous results found")

print("\nNew strategy: Buy after 3:30 PM, Sell next day at 3:00 PM")
print("="*60)

# Simulate some example trades
example_trades = [
    {"symbol": "AAPL", "buy_time": "15:45", "buy_price": 100, "sell_time": "15:00 next day", "sell_price": 101},
    {"symbol": "MSFT", "buy_time": "15:50", "buy_price": 200, "sell_time": "15:00 next day", "sell_price": 198},
    {"symbol": "GOOGL", "buy_time": "15:35", "buy_price": 150, "sell_time": "15:00 next day", "sell_price": 153},
]

total_return = 0
for trade in example_trades:
    ret = (trade['sell_price'] - trade['buy_price']) / trade['buy_price'] * 100
    total_return += ret
    print(f"{trade['symbol']}: Buy at {trade['buy_time']} @ ${trade['buy_price']}, Sell at {trade['sell_time']} @ ${trade['sell_price']}, Return: {ret:.2f}%")

avg_return = total_return / len(example_trades)
print(f"\nAverage return per trade: {avg_return:.2f}%")

print("\nKey differences from 9:30 AM exit:")
print("1. Holding period: ~23.5 hours (vs ~18 hours)")
print("2. Exit during active trading (3 PM) vs morning volatility (9:30 AM)")
print("3. More time for positions to develop")
print("4. Capital tied up longer, but potentially better exit prices")

# Now run the actual backtest
print("\nRunning full backtest...")
from option_flow_backtest_v5_late_only import OptionFlowBacktestV5

backtest = OptionFlowBacktestV5(initial_capital=100000)
print(f"Trade start time: {backtest.trade_start_time}")
print(f"Exit time: Next day 3:00 PM")

# Run with limited data for testing
option_file = "option_data/merged_deduplicated_2025M1_M9.csv"
if os.path.exists(option_file):
    # Load first 50 signals for quick test
    df = pd.read_csv(option_file)
    df_subset = df.head(50)
    df_subset.to_csv('test_subset.csv', index=False)
    
    try:
        backtest.run_backtest('test_subset.csv')
        backtest.calculate_metrics()
    except Exception as e:
        print(f"Error in backtest: {e}")
    finally:
        if os.path.exists('test_subset.csv'):
            os.remove('test_subset.csv')
else:
    print(f"Option data file not found: {option_file}")
