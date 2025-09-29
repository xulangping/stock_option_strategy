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
import yfinance as yf

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
    
    # Get SPY data for comparison
    start_date = df['time'].iloc[0].strftime('%Y-%m-%d')
    end_date = df['time'].iloc[-1].strftime('%Y-%m-%d')
    
    print(f"Downloading SPY data from {start_date} to {end_date}...")
    spy = yf.Ticker('SPY')
    spy_data = spy.history(start=start_date, end=end_date)
    spy_prices = spy_data['Close']
    
    # Calculate SPY returns
    spy_initial = spy_prices.iloc[0]
    spy_returns = ((spy_prices / spy_initial - 1) * 100).values  # Convert to percentage
    spy_dates = spy_prices.index
    
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
    plt.figure(figsize=(14, 7))
    
    # Calculate strategy returns
    strategy_returns = [(eq / initial_capital - 1) * 100 for eq in equity_curve]  # Convert to percentage
    
    # Plot returns
    plt.plot(times, strategy_returns, color='#FF6B6B', linewidth=2.5, label='Strategy Return')
    plt.plot(spy_dates, spy_returns, color='#4ECDC4', linewidth=2.5, label='SPY Return')
    
    # Add zero line for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Format
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.title('V5 Strategy vs SPY - Returns Comparison\n', fontsize=14, pad=15)
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:+.1f}%'))
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()  # Rotate dates
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=11)
    
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
    
    # Calculate SPY metrics
    spy_final_return = spy_returns[-1] if len(spy_returns) > 0 else 0
    excess_return = total_return - spy_final_return
    
    # Add statistics box
    stats_text = f'Initial Capital: ${initial_capital:,.0f}\n'
    stats_text += f'Final Equity: ${final_equity:,.0f}\n'
    stats_text += f'Strategy Return: {total_return:+.2f}%\n'
    stats_text += f'SPY Return: {spy_final_return:+.2f}%\n'
    stats_text += f'Excess Return: {excess_return:+.2f}%\n'
    stats_text += f'Max Drawdown: -{max_drawdown:.2f}%\n'
    stats_text += f'Min Equity: ${min_equity:,.0f}\n'
    stats_text += f'Max Equity: ${max_equity:,.0f}'
    
    # Add text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Add final return annotations
    plt.text(times[-1], strategy_returns[-1], f'  {strategy_returns[-1]:+.1f}%', 
             verticalalignment='center', fontsize=10, color='#FF6B6B', weight='bold')
    plt.text(spy_dates.tolist()[-1], spy_returns[-1], f'  {spy_returns[-1]:+.1f}%', 
             verticalalignment='center', fontsize=10, color='#4ECDC4', weight='bold')
    
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
    print(f"\nPerformance comparison:")
    print(f"Strategy Return: {total_return:+.2f}%")
    print(f"SPY Return: {spy_final_return:+.2f}%")
    print(f"Excess Return: {excess_return:+.2f}%")
    
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
