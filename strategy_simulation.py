#!/usr/bin/env python3
"""
Trading Strategy Monte Carlo Simulation
策略参数：胜率20%，赔率5倍，每天一次交易，一年200个交易日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import json
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StrategySimulator:
    def __init__(self, win_rate=0.2, payout_ratio=5.0, trading_days=200, initial_capital=100000):
        """
        Initialize strategy simulator
        
        Args:
            win_rate: 胜率 (default: 0.2 = 20%)
            payout_ratio: 赔率，赢的时候收益倍数 (default: 5.0 = 5倍)
            trading_days: 每年交易天数 (default: 200)
            initial_capital: 初始资金 (default: 100000)
        """
        self.win_rate = win_rate
        self.payout_ratio = payout_ratio
        self.trading_days = trading_days
        self.initial_capital = initial_capital
        
        # 计算期望收益
        self.expected_return_per_trade = (win_rate * payout_ratio) + ((1 - win_rate) * (-1))
        
        print(f"Strategy Parameters:")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Payout Ratio: {payout_ratio}x")
        print(f"Trading Days per Year: {trading_days}")
        print(f"Expected Return per Trade: {self.expected_return_per_trade:.3f}")
        print(f"Expected Annual Return: {self.expected_return_per_trade * trading_days:.2f}")
        print("-" * 50)
    
    def simulate_single_year(self, position_size_pct=0.02):
        """
        Simulate a single year of trading
        
        Args:
            position_size_pct: 每次交易占总资金的比例 (default: 0.02 = 2%)
        
        Returns:
            dict: Contains daily equity curve, returns, max drawdown, etc.
        """
        equity_curve = [self.initial_capital]
        daily_returns = []
        trades = []
        
        current_capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        
        for day in range(self.trading_days):
            # 确定交易金额（固定比例）
            position_size = current_capital * position_size_pct
            
            # 生成随机交易结果
            is_win = np.random.random() < self.win_rate
            
            if is_win:
                # 赢的情况：获得payout_ratio倍收益
                profit = position_size * self.payout_ratio
                trade_return = self.payout_ratio
            else:
                # 输的情况：损失全部头寸
                profit = -position_size
                trade_return = -1.0
            
            # 更新资金
            current_capital += profit
            equity_curve.append(current_capital)
            
            # 计算当日收益率
            daily_return = profit / (current_capital - profit)
            daily_returns.append(daily_return)
            
            # 记录交易
            trades.append({
                'day': day + 1,
                'is_win': is_win,
                'position_size': position_size,
                'profit': profit,
                'return': trade_return,
                'capital_after': current_capital
            })
            
            # 更新最大回撤
            if current_capital > peak_capital:
                peak_capital = current_capital
            
            current_drawdown = (peak_capital - current_capital) / peak_capital
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # 计算年化指标
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        annualized_return = total_return  # 已经是一年的数据
        
        # 计算胜率统计
        wins = sum(1 for t in trades if t['is_win'])
        actual_win_rate = wins / len(trades)
        
        return {
            'equity_curve': equity_curve,
            'daily_returns': daily_returns,
            'trades': trades,
            'final_capital': current_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'actual_win_rate': actual_win_rate,
            'total_trades': len(trades),
            'winning_trades': wins,
            'losing_trades': len(trades) - wins
        }
    
    def monte_carlo_simulation(self, num_simulations=1000, position_size_pct=0.02):
        """
        Run Monte Carlo simulation
        
        Args:
            num_simulations: 模拟次数
            position_size_pct: 每次交易占总资金的比例
        
        Returns:
            dict: Simulation results and statistics
        """
        print(f"Running {num_simulations} Monte Carlo simulations...")
        print(f"Position Size: {position_size_pct*100:.1f}% of capital per trade")
        print("-" * 50)
        
        results = []
        
        for i in range(num_simulations):
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_simulations} simulations...")
            
            result = self.simulate_single_year(position_size_pct)
            results.append(result)
        
        # 统计分析
        annual_returns = [r['annualized_return'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        final_capitals = [r['final_capital'] for r in results]
        actual_win_rates = [r['actual_win_rate'] for r in results]
        
        stats = {
            'num_simulations': num_simulations,
            'position_size_pct': position_size_pct,
            'annual_returns': {
                'mean': np.mean(annual_returns),
                'median': np.median(annual_returns),
                'std': np.std(annual_returns),
                'min': np.min(annual_returns),
                'max': np.max(annual_returns),
                'percentile_5': np.percentile(annual_returns, 5),
                'percentile_95': np.percentile(annual_returns, 95),
                'positive_years_pct': sum(1 for r in annual_returns if r > 0) / len(annual_returns) * 100
            },
            'max_drawdowns': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'percentile_5': np.percentile(max_drawdowns, 5),
                'percentile_95': np.percentile(max_drawdowns, 95)
            },
            'final_capitals': {
                'mean': np.mean(final_capitals),
                'median': np.median(final_capitals),
                'std': np.std(final_capitals),
                'min': np.min(final_capitals),
                'max': np.max(final_capitals)
            },
            'actual_win_rates': {
                'mean': np.mean(actual_win_rates),
                'std': np.std(actual_win_rates)
            }
        }
        
        return {
            'results': results,
            'stats': stats,
            'raw_data': {
                'annual_returns': annual_returns,
                'max_drawdowns': max_drawdowns,
                'final_capitals': final_capitals
            }
        }
    
    def create_visualizations(self, mc_results, save_path='simulation_results'):
        """
        Create visualization charts
        """
        results = mc_results['results']
        stats = mc_results['stats']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Trading Strategy Simulation Results\n'
                    f'Win Rate: {self.win_rate*100:.1f}%, Payout: {self.payout_ratio}x, '
                    f'Position Size: {stats["position_size_pct"]*100:.1f}%', 
                    fontsize=14, fontweight='bold')
        
        # 1. 年化收益率分布
        ax1 = axes[0, 0]
        annual_returns = mc_results['raw_data']['annual_returns']
        ax1.hist(annual_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(stats['annual_returns']['mean'], color='red', linestyle='--', 
                   label=f'Mean: {stats["annual_returns"]["mean"]:.1%}')
        ax1.axvline(stats['annual_returns']['median'], color='green', linestyle='--', 
                   label=f'Median: {stats["annual_returns"]["median"]:.1%}')
        ax1.set_xlabel('Annual Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Annual Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最大回撤分布
        ax2 = axes[0, 1]
        max_drawdowns = mc_results['raw_data']['max_drawdowns']
        ax2.hist(max_drawdowns, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(stats['max_drawdowns']['mean'], color='blue', linestyle='--', 
                   label=f'Mean: {stats["max_drawdowns"]["mean"]:.1%}')
        ax2.axvline(stats['max_drawdowns']['median'], color='green', linestyle='--', 
                   label=f'Median: {stats["max_drawdowns"]["median"]:.1%}')
        ax2.set_xlabel('Max Drawdown')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Maximum Drawdown Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 资金曲线示例（展示前10个模拟）
        ax3 = axes[1, 0]
        for i in range(min(10, len(results))):
            equity_curve = results[i]['equity_curve']
            days = list(range(len(equity_curve)))
            ax3.plot(days, equity_curve, alpha=0.6, linewidth=1)
        
        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Capital')
        ax3.set_title('Sample Equity Curves (First 10 Simulations)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 收益 vs 回撤散点图
        ax4 = axes[1, 1]
        ax4.scatter(max_drawdowns, annual_returns, alpha=0.6, s=10)
        ax4.set_xlabel('Max Drawdown')
        ax4.set_ylabel('Annual Return')
        ax4.set_title('Return vs Drawdown')
        ax4.grid(True, alpha=0.3)
        
        # 添加零线
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        Path(save_path).mkdir(exist_ok=True)
        plt.savefig(f'{save_path}/simulation_charts.png', dpi=300, bbox_inches='tight')
        print(f"Charts saved to {save_path}/simulation_charts.png")
        
        plt.show()
    
    def print_detailed_results(self, mc_results):
        """
        Print detailed simulation results
        """
        stats = mc_results['stats']
        
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*60)
        
        print(f"\nSimulation Settings:")
        print(f"  Number of Simulations: {stats['num_simulations']:,}")
        print(f"  Position Size per Trade: {stats['position_size_pct']*100:.1f}% of capital")
        print(f"  Initial Capital: ${self.initial_capital:,}")
        
        print(f"\nStrategy Parameters:")
        print(f"  Win Rate: {self.win_rate*100:.1f}%")
        print(f"  Payout Ratio: {self.payout_ratio}x")
        print(f"  Trading Days per Year: {self.trading_days}")
        print(f"  Expected Return per Trade: {self.expected_return_per_trade:.3f}")
        
        print(f"\nANNUAL RETURN STATISTICS:")
        ar = stats['annual_returns']
        print(f"  Mean Annual Return: {ar['mean']:.2%}")
        print(f"  Median Annual Return: {ar['median']:.2%}")
        print(f"  Standard Deviation: {ar['std']:.2%}")
        print(f"  Best Year: {ar['max']:.2%}")
        print(f"  Worst Year: {ar['min']:.2%}")
        print(f"  5th Percentile: {ar['percentile_5']:.2%}")
        print(f"  95th Percentile: {ar['percentile_95']:.2%}")
        print(f"  Profitable Years: {ar['positive_years_pct']:.1f}%")
        
        print(f"\nMAXIMUM DRAWDOWN STATISTICS:")
        dd = stats['max_drawdowns']
        print(f"  Mean Max Drawdown: {dd['mean']:.2%}")
        print(f"  Median Max Drawdown: {dd['median']:.2%}")
        print(f"  Standard Deviation: {dd['std']:.2%}")
        print(f"  Best Case (Min DD): {dd['min']:.2%}")
        print(f"  Worst Case (Max DD): {dd['max']:.2%}")
        print(f"  5th Percentile: {dd['percentile_5']:.2%}")
        print(f"  95th Percentile: {dd['percentile_95']:.2%}")
        
        print(f"\nFINAL CAPITAL STATISTICS:")
        fc = stats['final_capitals']
        print(f"  Mean Final Capital: ${fc['mean']:,.0f}")
        print(f"  Median Final Capital: ${fc['median']:,.0f}")
        print(f"  Min Final Capital: ${fc['min']:,.0f}")
        print(f"  Max Final Capital: ${fc['max']:,.0f}")
        
        print(f"\nACTUAL WIN RATE:")
        wr = stats['actual_win_rates']
        print(f"  Mean Actual Win Rate: {wr['mean']:.2%}")
        print(f"  Standard Deviation: {wr['std']:.2%}")
        
        # Risk-Reward Analysis
        print(f"\nRISK-REWARD ANALYSIS:")
        sharpe_approx = ar['mean'] / ar['std'] if ar['std'] > 0 else 0
        print(f"  Sharpe Ratio (approx): {sharpe_approx:.2f}")
        print(f"  Return/Drawdown Ratio: {ar['mean'] / dd['mean']:.2f}")
        
        # Probability Analysis
        prob_loss_10 = sum(1 for r in mc_results['raw_data']['annual_returns'] if r < -0.1) / len(mc_results['raw_data']['annual_returns']) * 100
        prob_gain_50 = sum(1 for r in mc_results['raw_data']['annual_returns'] if r > 0.5) / len(mc_results['raw_data']['annual_returns']) * 100
        
        print(f"\nPROBABILITY ANALYSIS:")
        print(f"  Probability of losing >10%: {prob_loss_10:.1f}%")
        print(f"  Probability of gaining >50%: {prob_gain_50:.1f}%")
        
    def save_results(self, mc_results, save_path='simulation_results'):
        """
        Save detailed results to JSON file
        """
        Path(save_path).mkdir(exist_ok=True)
        
        # 准备保存数据（移除不能序列化的部分）
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy_params': {
                'win_rate': self.win_rate,
                'payout_ratio': self.payout_ratio,
                'trading_days': self.trading_days,
                'initial_capital': self.initial_capital,
                'expected_return_per_trade': self.expected_return_per_trade
            },
            'simulation_stats': mc_results['stats'],
            'sample_results': mc_results['results'][:10]  # 只保存前10个模拟的详细结果
        }
        
        filename = f'{save_path}/simulation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to {filename}")

def main():
    """
    Main function to run the simulation
    """
    # 创建模拟器实例
    simulator = StrategySimulator(
        win_rate=0.2,        # 20% 胜率
        payout_ratio=5.0,    # 5倍赔率
        trading_days=200,    # 每年200个交易日
        initial_capital=100000  # 10万初始资金
    )
    
    # 运行蒙特卡罗模拟
    mc_results = simulator.monte_carlo_simulation(
        num_simulations=1000,    # 1000次模拟
        position_size_pct=0.02   # 每次交易2%资金
    )
    
    # 打印详细结果
    simulator.print_detailed_results(mc_results)
    
    # 创建可视化图表
    simulator.create_visualizations(mc_results)
    
    # 保存结果
    simulator.save_results(mc_results)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main() 