#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot backtest results - Fixed equity curve calculation
Optimized version with configurable parameters

使用方法：
    1. 修改下面的 CONFIGURATION 部分的参数
    2. 运行: python plot_v5_fixed.py
    
示例：
    # 绘制V6版本的回测结果
    TRADES_FILE = 'backtest_trades_v6_event_driven.csv'
    OUTPUT_IMAGE = 'v6_equity_curve.png'
    
    # 绘制V5版本的回测结果
    TRADES_FILE = 'backtest_trades_v5_late_only.csv'
    OUTPUT_IMAGE = 'v5_equity_curve.png'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import yfinance as yf
import os

# ============================================================================
# CONFIGURATION - 修改这里的参数
# ============================================================================

# 交易记录CSV文件路径（修改这里来绘制不同的回测结果）
TRADES_FILE = 'backtest_trades_v6_ma_filter.csv'

# 输出图片文件名
OUTPUT_IMAGE = 'equity_curve.png'

# 初始资金
INITIAL_CAPITAL = 100000

# SPY历史数据文件
SPY_DATA_FILE = 'spy_historical_data.csv'

# ============================================================================

# 设置matplotlib参数
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def download_and_save_spy_data(start_date=None, end_date=None, filename=SPY_DATA_FILE):
    """Download SPY historical data and save to CSV file"""
    print(f"Downloading SPY data from {start_date} to {end_date}...")
    
    # Set default date range if not provided
    if start_date is None:
        start_date = '2023-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    spy = yf.Ticker('SPY')
    spy_data = spy.history(start=start_date, end=end_date)
    
    # Save to CSV
    spy_data.to_csv(filename)
    print(f"SPY data saved to {filename}")
    print(f"Data range: {spy_data.index[0].strftime('%Y-%m-%d')} to {spy_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total records: {len(spy_data)}")
    
    return spy_data

def load_spy_data(filename=SPY_DATA_FILE):
    """Load SPY data from CSV file"""
    if not os.path.exists(filename):
        return None
    
    print(f"Loading SPY data from {filename}...")
    spy_data = pd.read_csv(filename, index_col=0, parse_dates=True)
    
    # Convert index to DatetimeIndex if it's not already
    if not isinstance(spy_data.index, pd.DatetimeIndex):
        spy_data.index = pd.to_datetime(spy_data.index, utc=True).tz_localize(None)
    elif spy_data.index.tz is not None:
        # Remove timezone info if present
        spy_data.index = spy_data.index.tz_localize(None)
    
    return spy_data

def get_spy_data(start_date, end_date, force_download=False):
    """Get SPY data, either from local file or download if necessary"""
    
    # Convert string dates to datetime (ensure timezone-naive)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Remove timezone if present
    if hasattr(start_dt, 'tz') and start_dt.tz is not None:
        start_dt = start_dt.tz_localize(None)
    if hasattr(end_dt, 'tz') and end_dt.tz is not None:
        end_dt = end_dt.tz_localize(None)
    
    # Check if we need to download
    need_download = force_download
    
    if not force_download:
        # Try to load existing data (already timezone-naive from load_spy_data)
        spy_data = load_spy_data()
        
        if spy_data is None:
            print("No local SPY data found, downloading...")
            need_download = True
        else:
            # Check if date range is sufficient
            data_start = spy_data.index[0]
            data_end = spy_data.index[-1]
            
            if data_start > start_dt or data_end < end_dt:
                print(f"Local data range ({data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}) " 
                      f"doesn't cover required range ({start_date} to {end_date})")
                print("Downloading updated data...")
                need_download = True
            else:
                print(f"Using local SPY data (covers {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')})")
    
    if need_download:
        # Download with some buffer
        buffer_start = (start_dt - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        buffer_end = (end_dt + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        spy_data = download_and_save_spy_data(buffer_start, buffer_end)
        
        # Ensure downloaded data is also timezone-naive
        if isinstance(spy_data.index, pd.DatetimeIndex) and spy_data.index.tz is not None:
            spy_data.index = spy_data.index.tz_localize(None)
    
    # Filter to requested range
    spy_data_filtered = spy_data[(spy_data.index >= start_dt) & (spy_data.index <= end_dt)]
    return spy_data_filtered['Close']

def plot_equity_curve():
    """Plot equity curve with correct calculation"""
    print(f"\nPlotting equity curve from: {TRADES_FILE}")
    print(f"Output will be saved to: {OUTPUT_IMAGE}\n")
    
    # Check if file exists
    if not os.path.exists(TRADES_FILE):
        print(f"Error: Cannot find {TRADES_FILE}")
        return
    
    # Read trades data
    df = pd.read_csv(TRADES_FILE)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Get SPY data for comparison
    start_date = df['time'].iloc[0].strftime('%Y-%m-%d')
    end_date = df['time'].iloc[-1].strftime('%Y-%m-%d')
    
    # Use new function to get SPY data (from local or download)
    spy_prices = get_spy_data(start_date, end_date)
    
    # Calculate SPY returns
    spy_initial = spy_prices.iloc[0]
    spy_returns = ((spy_prices / spy_initial - 1) * 100).values  # Convert to percentage
    spy_dates = spy_prices.index
    
    # Initialize tracking variables
    cash = INITIAL_CAPITAL
    positions = {}  # Track open positions
    equity_curve = [INITIAL_CAPITAL]
    times = [df['time'].iloc[0].replace(hour=0, minute=0, second=0)]
    
    # Process each trade
    for idx, row in df.iterrows():
        symbol = row['symbol']
        action = row['action']
        
        if action == 'BUY':
            # Buy: cash decreases, position opens
            cash -= row['net_value']
            # Support multiple positions in same symbol (though not ideal)
            if symbol not in positions:
                positions[symbol] = {
                    'count': 0,
                    'net_value_paid': 0
                }
            positions[symbol]['count'] += 1
            positions[symbol]['net_value_paid'] += row['net_value']
        else:
            # Sell: cash increases, position closes
            cash += row['net_value']
            if symbol in positions:
                positions[symbol]['count'] -= 1
                positions[symbol]['net_value_paid'] -= row['net_value']
                # Remove if all positions closed
                if positions[symbol]['count'] <= 0:
                    del positions[symbol]
        
        # Calculate total equity = cash + what we paid for open positions
        # This represents our total capital at any point in time
        position_value = sum(pos['net_value_paid'] for pos in positions.values())
        total_equity = cash + position_value
        
        equity_curve.append(total_equity)
        times.append(row['time'])
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Calculate strategy returns
    strategy_returns = [(eq / INITIAL_CAPITAL - 1) * 100 for eq in equity_curve]  # Convert to percentage
    
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
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
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
    stats_text = f'Initial Capital: ${INITIAL_CAPITAL:,.0f}\n'
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
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {OUTPUT_IMAGE}")
    
    # Print some debug info
    print(f"\nEquity progression:")
    print(f"Initial: ${INITIAL_CAPITAL:,.2f}")
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
    print("="*70)
    print("Equity Curve Plotter - Optimized Version")
    print("="*70)
    print(f"Configuration:")
    print(f"  Trades file: {TRADES_FILE}")
    print(f"  Output image: {OUTPUT_IMAGE}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,}")
    print("="*70)
    
    # Plot equity curve
    plot_equity_curve()

if __name__ == "__main__":
    main()
