#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot V5 backtest results - Fixed equity curve calculation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 设置matplotlib参数
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_equity_curve(trades_file='backtest_trades_v5_late_only.csv', initial_capital=100000):
    """Plot equity curve with correct calculation"""
    # Read trades data
    df = pd.read_csv(trades_file)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Initialize tracking variables
    cash = initial_capital
    positions = {}  # Track open positions
    equity_curve = [initial_capital]
    times = [df['time'].iloc[0].replace(hour=0, minute=0, second=0)]
    
    # Process each trade
    for idx, row in df.iterrows():
        symbol = row['symbol']
        action = row['action']
        
        if action == 'BUY':
            # Buy: cash decreases, position opens
            cash -= row['net_value']
            positions[symbol] = {
                'shares': row['shares'],
                'entry_price': row['price'],
                'entry_value': row['net_value']
            }
        else:
            # Sell: cash increases, position closes
            cash += row['net_value']
            if symbol in positions:
                del positions[symbol]
        
        # Calculate total equity = cash + value of open positions
        position_value = sum(pos['entry_value'] for pos in positions.values())
        total_equity = cash + position_value
        
        equity_curve.append(total_equity)
        times.append(row['time'])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot equity curve
    plt.plot(times, equity_curve, color='#2E86AB', linewidth=2.5)
    
    # Add initial capital line
    plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label=f'Initial: ${initial_capital:,.0f}')
    
    # Fill area under curve
    equity_array = np.array(equity_curve)
    times_array = np.array(times)
    plt.fill_between(times_array, equity_array, initial_capital, 
                     where=(equity_array >= initial_capital), 
                     color='green', alpha=0.1, interpolate=True)
    plt.fill_between(times_array, equity_array, initial_capital, 
                     where=(equity_array < initial_capital), 
                     color='red', alpha=0.1, interpolate=True)
    
    # Format
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Equity ($)', fontsize=12)
    plt.title('V5 Strategy - Total Equity Over Time\n(Trading Only After 15:45)', fontsize=14, pad=15)
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()  # Rotate dates
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Calculate final metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    max_equity = max(equity_curve)
    min_equity = min(equity_curve)
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        drawdown = (peak - eq) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Add statistics box
    stats_text = f'Initial Capital: ${initial_capital:,.0f}\n'
    stats_text += f'Final Equity: ${final_equity:,.0f}\n'
    stats_text += f'Total Return: {total_return:+.2f}%\n'
    stats_text += f'Max Drawdown: -{max_drawdown:.2f}%\n'
    stats_text += f'Min Equity: ${min_equity:,.0f}\n'
    stats_text += f'Max Equity: ${max_equity:,.0f}'
    
    # Add text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Add final value annotation
    plt.text(times[-1], final_equity, f'  ${final_equity:,.0f}\n  ({total_return:+.1f}%)', 
             verticalalignment='center', fontsize=10, color='darkblue', weight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('v5_equity_curve_fixed.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved to: v5_equity_curve_fixed.png")
    
    # Print some debug info
    print(f"\nEquity progression:")
    print(f"Initial: ${initial_capital:,.2f}")
    print(f"Min: ${min_equity:,.2f}")
    print(f"Max: ${max_equity:,.2f}")
    print(f"Final: ${final_equity:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Show plot
    plt.show()

def main():
    """Main function"""
    print("Plotting V5 Strategy Equity Curve (Fixed)...")
    
    # Check if file exists
    import os
    if os.path.exists('backtest_trades_v5_late_only.csv'):
        plot_equity_curve()
    else:
        print("Error: Cannot find backtest_trades_v5_late_only.csv")

if __name__ == "__main__":
    main()
