#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the final changes to the backtest
"""

import pandas as pd
from option_flow_backtest_v2 import OptionFlowBacktestV2

# Load the data to check premium values
df = pd.read_csv("option_data/flow_true_mid_side,no_side,bid_side_false_true_Common Stock_50_60_0.8_100000_5_Tue Sep 16 2025 00_00_00 GMT+0800 (中国标准时间)_Fri Sep 19 2025 23_59_00 GMT+0800 (中国标准时间)_true_true_Calls_true_strategy_v1_1000000000000.csv")

print("Premium values and expected position sizes:")
print("="*60)

# Show first 10 signals with premium values
for idx, row in df.head(10).iterrows():
    premium = row['premium']
    expected_position = min(premium / 1000, 20)  # Cap at 20%
    print(f"{row['underlying_symbol']:6} | Premium: ${premium:>10,.0f} | Expected Position: {expected_position:>5.1%}")

print("\n" + "="*60)
print("\nRunning test backtest...")
print("="*60)

# Test with a small subset
backtest = OptionFlowBacktestV2(initial_capital=100000)

# Create a small test file with just a few signals
test_df = df.head(10)
test_df.to_csv('temp_test.csv', index=False)

# Run backtest
backtest.run_backtest('temp_test.csv')
backtest.calculate_metrics()

# Clean up
import os
os.remove('temp_test.csv')

print("\n" + "="*60)
print("Key things to verify:")
print("1. Position sizes match option premium / 1000")
print("2. Daily position usage doesn't exceed 80%")
print("3. Same stocks aren't bought within 5 days")
print("4. All exits happen during market hours (9:30 AM - 4:00 PM)")
print("="*60)
