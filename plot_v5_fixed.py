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
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import yfinance as yf
import os

# ============================================================================
# CONFIGURATION - 修改这里的参数
# ============================================================================

# 交易记录CSV文件路径（修改这里来绘制不同的回测结果）
TRADES_FILE = 'backtest_trades_v6_event_driven.csv'

# 输出图片文件名
OUTPUT_IMAGE = 'equity_curve_v6_event_driven.png'

# 初始资金
INITIAL_CAPITAL = 100000

# SPY和QQQ历史数据文件
SPY_DATA_FILE = 'spy_historical_data.csv'
QQQ_DATA_FILE = 'qqq_historical_data.csv'

# ============================================================================

# 设置matplotlib参数
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def download_and_save_data(symbol='SPY', start_date=None, end_date=None, filename=None):
    """Download historical data and save to CSV file"""
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Set default date range if not provided
    if start_date is None:
        start_date = '2023-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Save to CSV
    data.to_csv(filename)
    print(f"{symbol} data saved to {filename}")
    print(f"Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total records: {len(data)}")
    
    return data

def load_data(symbol='SPY', filename=SPY_DATA_FILE):
    """Load data from CSV file"""
    if not os.path.exists(filename):
        return None
    
    print(f"Loading {symbol} data from {filename}...")
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    
    # Convert index to DatetimeIndex if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    elif data.index.tz is not None:
        # Remove timezone info if present
        data.index = data.index.tz_localize(None)
    
    return data

def get_data(symbol='SPY', start_date=None, end_date=None, filename=SPY_DATA_FILE, force_download=False):
    """Get data, either from local file or download if necessary"""
    
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
        # Try to load existing data
        data = load_data(symbol, filename)
        
        if data is None:
            print(f"No local {symbol} data found, downloading...")
            need_download = True
        else:
            # Check if date range is sufficient
            data_start = data.index[0]
            data_end = data.index[-1]
            
            if data_start > start_dt or data_end < end_dt:
                print(f"Local data range ({data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}) " 
                      f"doesn't cover required range ({start_date} to {end_date})")
                print("Downloading updated data...")
                need_download = True
            else:
                print(f"Using local {symbol} data (covers {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')})")
    
    if need_download:
        # Download with some buffer
        buffer_start = (start_dt - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        buffer_end = (end_dt + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        data = download_and_save_data(symbol, buffer_start, buffer_end, filename)
        
        # Ensure downloaded data is also timezone-naive
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    
    # Filter to requested range
    data_filtered = data[(data.index >= start_dt) & (data.index <= end_dt)]
    return data_filtered['Close']

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
    
    # Get SPY and QQQ data for comparison
    start_date = df['time'].iloc[0].strftime('%Y-%m-%d')
    # Use a later end date to capture full period (add buffer for mark-to-market)
    end_date = (df['time'].iloc[-1] + pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    
    # Use new function to get SPY data (from local or download)
    spy_prices = get_data('SPY', start_date, end_date, SPY_DATA_FILE)
    
    # Get QQQ data
    qqq_prices = get_data('QQQ', start_date, end_date, QQQ_DATA_FILE)
    
    # Calculate SPY returns
    spy_initial = spy_prices.iloc[0]
    spy_returns = ((spy_prices / spy_initial - 1) * 100).values  # Convert to percentage
    spy_dates = spy_prices.index
    
    # Calculate QQQ returns
    qqq_initial = qqq_prices.iloc[0]
    qqq_returns = ((qqq_prices / qqq_initial - 1) * 100).values  # Convert to percentage
    qqq_dates = qqq_prices.index
    
    # Initialize tracking variables
    cash = INITIAL_CAPITAL
    positions = {}  # Track open positions: {symbol: {'shares': int, 'avg_cost': float}}
    
    # Process each trade to build position history
    trade_idx = 0
    
    # Create daily equity curve by marking to market every day
    equity_curve = []
    times = []
    
    # Get date range for daily calculation
    start_dt = qqq_prices.index[0]
    end_dt = qqq_prices.index[-1]
    
    print(f"\nCalculating daily equity curve from {start_dt.date()} to {end_dt.date()}...")
    
    # Iterate through each trading day
    for current_date in qqq_prices.index:
        # Process all trades that occurred on or before this date
        while trade_idx < len(df) and df.iloc[trade_idx]['time'] <= current_date:
            row = df.iloc[trade_idx]
            symbol = row['symbol']
            action = row['action']
            shares = row['shares']
            
            if action in ['BUY', 'ROTATION_BUY_BACK']:
                cash -= row['net_value']
                if symbol not in positions:
                    positions[symbol] = {'shares': 0, 'total_cost': 0}
                positions[symbol]['shares'] += shares
                positions[symbol]['total_cost'] += row['net_value']
            else:  # Sell actions
                cash += row['net_value']
                if symbol in positions:
                    avg_cost_per_share = positions[symbol]['total_cost'] / positions[symbol]['shares'] if positions[symbol]['shares'] > 0 else 0
                    sold_cost = avg_cost_per_share * shares
                    positions[symbol]['shares'] -= shares
                    positions[symbol]['total_cost'] -= sold_cost
                    if positions[symbol]['shares'] <= 0:
                        del positions[symbol]
            
            trade_idx += 1
        
        # Calculate equity using current market prices
        total_equity = cash
        
        for symbol, pos_info in positions.items():
            if symbol == 'QQQ':
                # Get QQQ price at current_date
                if current_date in qqq_prices.index:
                    current_price = qqq_prices.loc[current_date]
                else:
                    # Use last available price before current_date
                    past_prices = qqq_prices[qqq_prices.index <= current_date]
                    current_price = past_prices.iloc[-1] if len(past_prices) > 0 else pos_info['total_cost'] / pos_info['shares']
                
                market_value = pos_info['shares'] * current_price
                total_equity += market_value
            else:
                # For other symbols, use cost basis (we don't have historical prices)
                total_equity += pos_info['total_cost']
        
        equity_curve.append(total_equity)
        times.append(current_date)
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Calculate strategy returns
    strategy_returns = [(eq / INITIAL_CAPITAL - 1) * 100 for eq in equity_curve]  # Convert to percentage
    
    # Plot returns
    plt.plot(times, strategy_returns, color='#FF6B6B', linewidth=2.5, label='Strategy Return')
    plt.plot(spy_dates, spy_returns, color='#4ECDC4', linewidth=2.5, label='SPY Return')
    plt.plot(qqq_dates, qqq_returns, color='#95E1D3', linewidth=2.5, label='QQQ Return')
    
    # Add zero line for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Format
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.title('V7 QQQ Rotation Strategy vs SPY vs QQQ - Returns Comparison\n', fontsize=14, pad=15)
    
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
    
    # Calculate SPY and QQQ metrics
    spy_final_return = spy_returns[-1] if len(spy_returns) > 0 else 0
    qqq_final_return = qqq_returns[-1] if len(qqq_returns) > 0 else 0
    excess_return_spy = total_return - spy_final_return
    excess_return_qqq = total_return - qqq_final_return
    
    # Calculate yearly returns
    yearly_returns = {}
    yearly_spy_returns = {}
    yearly_qqq_returns = {}
    
    # Create DataFrame for easier yearly analysis
    equity_df = pd.DataFrame({'time': times, 'equity': equity_curve})
    equity_df['year'] = equity_df['time'].dt.year
    
    # Get the first year of data
    first_year = equity_df['year'].iloc[0]
    
    # Calculate returns for each year
    for year in [2023, 2024, 2025]:
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) > 0:
            # Determine start equity for the year
            if year == first_year:
                # For the first year, use the very first equity value (INITIAL_CAPITAL)
                start_equity = equity_df.iloc[0]['equity']
            else:
                # For subsequent years, use Dec 31 of previous year
                # Find the last equity value on or before Dec 31 of previous year
                year_start_date = pd.Timestamp(f'{year}-01-01')
                prev_data = equity_df[equity_df['time'] < year_start_date]
                if len(prev_data) > 0:
                    start_equity = prev_data.iloc[-1]['equity']
                else:
                    # If no data before this year, use first equity of the year
                    start_equity = year_data.iloc[0]['equity']
            
            # End equity is the last equity value in the year
            end_equity = year_data.iloc[-1]['equity']
            
            yearly_return = (end_equity - start_equity) / start_equity * 100
            yearly_returns[year] = yearly_return
            
            # Calculate SPY and QQQ returns for the same period
            # Use the same date range as equity calculation
            if year == first_year:
                start_date = equity_df.iloc[0]['time']
            else:
                start_date = pd.Timestamp(f'{year}-01-01')
            
            end_date = year_data.iloc[-1]['time']
            
            # Get SPY price at start and end
            spy_start_data = spy_prices[spy_prices.index <= start_date]
            if len(spy_start_data) > 0:
                spy_start_price = spy_start_data.iloc[-1]
            else:
                spy_start_price = spy_prices.iloc[0]
            
            spy_end_data = spy_prices[spy_prices.index <= end_date]
            if len(spy_end_data) > 0:
                spy_end_price = spy_end_data.iloc[-1]
                spy_yearly_return = (spy_end_price - spy_start_price) / spy_start_price * 100
                yearly_spy_returns[year] = spy_yearly_return
            
            # Get QQQ price at start and end
            qqq_start_data = qqq_prices[qqq_prices.index <= start_date]
            if len(qqq_start_data) > 0:
                qqq_start_price = qqq_start_data.iloc[-1]
            else:
                qqq_start_price = qqq_prices.iloc[0]
            
            qqq_end_data = qqq_prices[qqq_prices.index <= end_date]
            if len(qqq_end_data) > 0:
                qqq_end_price = qqq_end_data.iloc[-1]
                qqq_yearly_return = (qqq_end_price - qqq_start_price) / qqq_start_price * 100
                yearly_qqq_returns[year] = qqq_yearly_return
    
    # Add statistics box
    stats_text = f'Initial Capital: ${INITIAL_CAPITAL:,.0f}\n'
    stats_text += f'Final Equity: ${final_equity:,.0f}\n'
    stats_text += f'Strategy Return: {total_return:+.2f}%\n'
    stats_text += f'SPY Return: {spy_final_return:+.2f}%\n'
    stats_text += f'QQQ Return: {qqq_final_return:+.2f}%\n'
    stats_text += f'Excess vs SPY: {excess_return_spy:+.2f}%\n'
    stats_text += f'Excess vs QQQ: {excess_return_qqq:+.2f}%\n'
    stats_text += f'Max Drawdown: -{max_drawdown:.2f}%'
    
    # Add text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Add final return annotations
    plt.text(times[-1], strategy_returns[-1], f'  {strategy_returns[-1]:+.1f}%', 
             verticalalignment='center', fontsize=10, color='#FF6B6B', weight='bold')
    plt.text(spy_dates.tolist()[-1], spy_returns[-1], f'  {spy_returns[-1]:+.1f}%', 
             verticalalignment='center', fontsize=10, color='#4ECDC4', weight='bold')
    plt.text(qqq_dates.tolist()[-1], qqq_returns[-1], f'  {qqq_returns[-1]:+.1f}%', 
             verticalalignment='center', fontsize=10, color='#95E1D3', weight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {OUTPUT_IMAGE}")
    
    # Print some debug info
    print("\nEquity progression:")
    print(f"Initial: ${INITIAL_CAPITAL:,.2f}")
    print(f"Min: ${min_equity:,.2f}")
    print(f"Max: ${max_equity:,.2f}")
    print(f"Final: ${final_equity:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("\nPerformance comparison:")
    print(f"Strategy Return: {total_return:+.2f}%")
    print(f"SPY Return: {spy_final_return:+.2f}%")
    print(f"QQQ Return: {qqq_final_return:+.2f}%")
    print(f"Excess vs SPY: {excess_return_spy:+.2f}%")
    print(f"Excess vs QQQ: {excess_return_qqq:+.2f}%")
    
    # Print yearly returns
    print("\nYearly Returns:")
    print("-" * 100)
    print(f"{'Year':<6} {'Start Equity':<15} {'End Equity':<15} {'Strategy':<11} {'SPY':<11} {'QQQ':<11} {'vs SPY':<11} {'vs QQQ':<11}")
    print("-" * 100)
    
    for year in [2023, 2024, 2025]:
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) > 0:
            # Get start equity (same logic as in calculation)
            if year == first_year:
                start_eq = equity_df.iloc[0]['equity']
            else:
                year_start_date = pd.Timestamp(f'{year}-01-01')
                prev_data = equity_df[equity_df['time'] < year_start_date]
                if len(prev_data) > 0:
                    start_eq = prev_data.iloc[-1]['equity']
                else:
                    start_eq = year_data.iloc[0]['equity']
            
            end_eq = year_data.iloc[-1]['equity']
            
            if year in yearly_returns:
                strategy_ret = yearly_returns[year]
                spy_ret = yearly_spy_returns.get(year, 0)
                qqq_ret = yearly_qqq_returns.get(year, 0)
                excess_spy = strategy_ret - spy_ret
                excess_qqq = strategy_ret - qqq_ret
                print(f"{year:<6} ${start_eq:>13,.0f}  ${end_eq:>13,.0f}  {strategy_ret:>+9.2f}%  {spy_ret:>+9.2f}%  {qqq_ret:>+9.2f}%  {excess_spy:>+9.2f}%  {excess_qqq:>+9.2f}%")
        else:
            print(f"{year:<6} {'N/A':<15} {'N/A':<15} {'N/A':<11} {'N/A':<11} {'N/A':<11} {'N/A':<11} {'N/A':<11}")
    print("-" * 100)
    
    # Show plot
    plt.show()

def main():
    """Main function"""
    print("="*70)
    print("Equity Curve Plotter - Optimized Version")
    print("="*70)
    print("Configuration:")
    print(f"  Trades file: {TRADES_FILE}")
    print(f"  Output image: {OUTPUT_IMAGE}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,}")
    print("="*70)
    
    # Plot equity curve
    plot_equity_curve()

if __name__ == "__main__":
    main()
