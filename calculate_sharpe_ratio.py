#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Sharpe Ratio and compare with S&P 500
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

def calculate_strategy_metrics(trades_file='backtest_trades_v5_late_only.csv'):
    """Calculate strategy returns and metrics"""
    # Read trades data
    df = pd.read_csv(trades_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Get date range
    start_date = df['time'].min()
    end_date = df['time'].max()
    
    # Calculate daily returns for the strategy
    df_sorted = df.sort_values('time')
    
    # Build equity curve
    initial_capital = 100000
    cash = initial_capital
    positions = {}
    
    equity_by_date = {}
    
    for idx, row in df_sorted.iterrows():
        date = row['time'].date()
        
        if row['action'] == 'BUY':
            cash -= row['net_value']
            positions[row['symbol']] = row['net_value']
        else:
            cash += row['net_value']
            if row['symbol'] in positions:
                del positions[row['symbol']]
        
        # Calculate total equity
        position_value = sum(positions.values())
        total_equity = cash + position_value
        equity_by_date[date] = total_equity
    
    # Convert to daily returns
    dates = sorted(equity_by_date.keys())
    equity_values = [initial_capital]  # Start with initial capital
    
    for date in dates:
        equity_values.append(equity_by_date[date])
    
    # Calculate daily returns
    equity_series = pd.Series(equity_values)
    daily_returns = equity_series.pct_change().dropna()
    
    # Get S&P 500 data for the same period
    print(f"Fetching S&P 500 data from {start_date.date()} to {end_date.date()}...")
    spy = yf.download('SPY', start=start_date.date(), end=end_date.date(), progress=False)
    
    # Handle different column formats
    if 'Adj Close' in spy.columns:
        spy_close = spy['Adj Close']
    elif 'Close' in spy.columns:
        spy_close = spy['Close']
    else:
        # Handle multi-level columns
        spy_close = spy[('Close', 'SPY')] if isinstance(spy.columns, pd.MultiIndex) else spy.iloc[:, -1]
    
    spy_returns = spy_close.pct_change().dropna()
    
    # Calculate metrics
    strategy_annual_return = (equity_values[-1] / initial_capital) ** (252 / len(dates)) - 1
    strategy_volatility = daily_returns.std() * np.sqrt(252)
    
    spy_total_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1)
    spy_annual_return = (1 + spy_total_return) ** (252 / len(spy)) - 1
    spy_volatility = spy_returns.std() * np.sqrt(252)
    
    # Risk-free rate (assume 4% annual for 2024-2025 period)
    risk_free_rate = 0.04
    
    # Calculate Sharpe ratios
    strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
    spy_sharpe = (spy_annual_return - risk_free_rate) / spy_volatility if spy_volatility > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\nTime Period: {start_date.date()} to {end_date.date()}")
    print(f"Trading Days: {len(dates)}")
    
    print(f"\nV5 Strategy (Late Trading Only):")
    print(f"  Total Return: {(equity_values[-1] / initial_capital - 1) * 100:.2f}%")
    print(f"  Annualized Return: {strategy_annual_return * 100:.2f}%")
    print(f"  Annualized Volatility: {strategy_volatility * 100:.2f}%")
    print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
    
    print(f"\nS&P 500 (SPY):")
    print(f"  Total Return: {spy_total_return * 100:.2f}%")
    print(f"  Annualized Return: {spy_annual_return * 100:.2f}%")
    print(f"  Annualized Volatility: {spy_volatility * 100:.2f}%")
    print(f"  Sharpe Ratio: {spy_sharpe:.2f}")
    
    print(f"\nRisk-Free Rate: {risk_free_rate * 100:.1f}%")
    
    print(f"\nOutperformance:")
    print(f"  Return Difference: {(strategy_annual_return - spy_annual_return) * 100:.2f}%")
    print(f"  Sharpe Difference: {strategy_sharpe - spy_sharpe:.2f}")
    
    # Calculate Information Ratio
    active_returns = []
    
    # Match dates between strategy and SPY
    for i in range(1, len(equity_values)):
        strategy_ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
        
        # Find corresponding SPY return
        if i-1 < len(dates) and dates[i-1] in spy.index:
            if i < len(dates) and dates[i] in spy.index:
                spy_ret = (spy_close.loc[dates[i]] - spy_close.loc[dates[i-1]]) / spy_close.loc[dates[i-1]]
                active_returns.append(strategy_ret - spy_ret)
    
    if active_returns:
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = (strategy_annual_return - spy_annual_return) / tracking_error if tracking_error > 0 else 0
        print(f"\nInformation Ratio: {information_ratio:.2f}")
        print(f"Tracking Error: {tracking_error * 100:.2f}%")

def main():
    """Main function"""
    import os
    
    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance")
        import yfinance
    
    if os.path.exists('backtest_trades_v5_late_only.csv'):
        calculate_strategy_metrics()
    else:
        print("Error: Cannot find backtest_trades_v5_late_only.csv")
        print("Please run option_flow_backtest_v5_late_only.py first")

if __name__ == "__main__":
    main()
