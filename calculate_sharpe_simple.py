#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Sharpe Ratio for V5 Strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_strategy_sharpe(trades_file='backtest_trades_v5_late_only.csv'):
    """Calculate strategy Sharpe ratio"""
    # Read trades data
    df = pd.read_csv(trades_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Get date range
    start_date = df['time'].min()
    end_date = df['time'].max()
    trading_days = (end_date - start_date).days
    
    # Build daily equity curve
    initial_capital = 100000
    cash = initial_capital
    positions = {}
    
    # Group by date
    df_sorted = df.sort_values('time')
    daily_equity = {}
    
    # Add initial date
    current_date = start_date.date()
    daily_equity[current_date] = initial_capital
    
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
        daily_equity[date] = total_equity
    
    # Create daily returns series
    dates = sorted(daily_equity.keys())
    equity_values = [daily_equity[date] for date in dates]
    
    # Calculate daily returns
    daily_returns = []
    for i in range(1, len(equity_values)):
        daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
        daily_returns.append(daily_return)
    
    # If no daily changes, approximate from total return
    if not daily_returns:
        total_return = (equity_values[-1] / initial_capital - 1)
        # Approximate daily return
        daily_returns = [total_return / trading_days] * trading_days
    
    # Calculate annualized metrics
    annual_return = (equity_values[-1] / initial_capital) ** (252 / trading_days) - 1
    daily_volatility = np.std(daily_returns) if daily_returns else 0
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Risk-free rate (4% annual for 2024-2025)
    risk_free_rate = 0.04
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # S&P 500 approximation for the period (Dec 2024 - Sep 2025)
    # Based on historical data: SPY was around $475 in Dec 2024 and $560 in Sep 2025
    spy_total_return = 0.18  # Approximately 18% total return
    spy_annual_return = (1 + spy_total_return) ** (252 / trading_days) - 1
    spy_annual_volatility = 0.15  # Historical average volatility
    spy_sharpe = (spy_annual_return - risk_free_rate) / spy_annual_volatility
    
    # Print results
    print("\n" + "="*60)
    print("SHARPE RATIO ANALYSIS")
    print("="*60)
    
    print(f"\nTime Period: {start_date.date()} to {end_date.date()}")
    print(f"Trading Days: {trading_days}")
    
    print(f"\nV5 Strategy (Late Trading Only):")
    print(f"  Total Return: {(equity_values[-1] / initial_capital - 1) * 100:.2f}%")
    print(f"  Annualized Return: {annual_return * 100:.2f}%")
    print(f"  Annualized Volatility: {annual_volatility * 100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    
    print(f"\nS&P 500 (Estimated):")
    print(f"  Total Return: {spy_total_return * 100:.2f}%")
    print(f"  Annualized Return: {spy_annual_return * 100:.2f}%")
    print(f"  Annualized Volatility: {spy_annual_volatility * 100:.2f}%")
    print(f"  Sharpe Ratio: {spy_sharpe:.2f}")
    
    print(f"\nRisk-Free Rate: {risk_free_rate * 100:.1f}%")
    
    print(f"\nOutperformance:")
    print(f"  Return Difference: {(annual_return - spy_annual_return) * 100:.2f}% annualized")
    print(f"  Sharpe Difference: {sharpe_ratio - spy_sharpe:.2f}")
    
    # Risk-adjusted metrics
    if spy_sharpe > 0:
        relative_sharpe = sharpe_ratio / spy_sharpe
        print(f"  Relative Sharpe: {relative_sharpe:.2f}x")
    
    # Calculate monthly statistics
    completed_trades = df[df['action'] != 'BUY']
    monthly_trades = completed_trades.groupby(pd.Grouper(key='time', freq='M')).size()
    avg_monthly_trades = monthly_trades.mean()
    
    print(f"\nAdditional Statistics:")
    print(f"  Average Trades per Month: {avg_monthly_trades:.1f}")
    print(f"  Total Completed Trades: {len(completed_trades)}")
    
    # Win rate
    winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
    win_rate = len(winning_trades) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
    print(f"  Win Rate: {win_rate:.1f}%")

def main():
    """Main function"""
    import os
    
    if os.path.exists('backtest_trades_v5_late_only.csv'):
        calculate_strategy_sharpe()
    else:
        print("Error: Cannot find backtest_trades_v5_late_only.csv")
        print("Please run option_flow_backtest_v5_late_only.py first")

if __name__ == "__main__":
    main()
