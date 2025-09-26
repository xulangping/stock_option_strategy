#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账户资产追踪器 - 每天记录账户总资产变化
可以设置定时任务每天运行一次
"""

from futu import *
import datetime
import json
import os
import pandas as pd
from pathlib import Path

# 配置
US_SIM_ACC_ID = 16428245
UNLOCK_PWD = '153811'
HISTORY_FILE = 'account_history.json'
CSV_FILE = 'account_history.csv'

class AccountTracker:
    def __init__(self):
        self.trd_ctx = None
        self.history = self.load_history()
        
    def connect(self):
        """连接富途"""
        self.trd_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host='127.0.0.1', 
            port=11111, 
            security_firm=SecurityFirm.FUTUSECURITIES
        )
        
        # 解锁
        ret, data = self.trd_ctx.unlock_trade(UNLOCK_PWD)
        if ret != RET_OK:
            print(f"解锁失败: {data}")
            return False
        return True
    
    def disconnect(self):
        """断开连接"""
        if self.trd_ctx:
            self.trd_ctx.close()
    
    def load_history(self):
        """加载历史记录"""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return {'records': [], 'initial_capital': 1000000}
    
    def save_history(self):
        """保存历史记录"""
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 同时保存为CSV便于分析
        if self.history['records']:
            df = pd.DataFrame(self.history['records'])
            df.to_csv(CSV_FILE, index=False)
            print(f"历史记录已保存到 {CSV_FILE}")
    
    def get_account_snapshot(self):
        """获取账户快照"""
        snapshot = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'datetime': datetime.datetime.now().isoformat(),
            'total_assets': 0,
            'cash': 0,
            'market_value': 0,
            'positions': [],
            'daily_pnl': 0,
            'daily_pnl_pct': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0
        }
        
        # 获取账户信息
        ret, acc_info = self.trd_ctx.accinfo_query(
            trd_env=TrdEnv.SIMULATE,
            acc_id=US_SIM_ACC_ID
        )
        
        if ret == RET_OK and not acc_info.empty:
            info = acc_info.iloc[0]
            snapshot['total_assets'] = float(info['total_assets'])
            snapshot['cash'] = float(info['cash'])
            snapshot['market_value'] = float(info['market_val'])
        
        # 获取持仓
        ret, positions = self.trd_ctx.position_list_query(
            trd_env=TrdEnv.SIMULATE,
            acc_id=US_SIM_ACC_ID
        )
        
        if ret == RET_OK and not positions.empty:
            for _, pos in positions.iterrows():
                position_info = {
                    'symbol': pos['code'],
                    'qty': int(pos['qty']),
                    'cost_price': float(pos['cost_price']),
                    'current_price': float(pos['nominal_price']),
                    'market_value': float(pos['market_val']),
                    'pnl': float(pos.get('pl_val', 0)),
                    'pnl_pct': float(pos.get('pl_ratio', 0))
                }
                snapshot['positions'].append(position_info)
        
        # 计算日盈亏（与昨天比较）
        if self.history['records']:
            last_record = self.history['records'][-1]
            last_total = last_record['total_assets']
            snapshot['daily_pnl'] = snapshot['total_assets'] - last_total
            snapshot['daily_pnl_pct'] = (snapshot['daily_pnl'] / last_total) * 100
        
        # 计算总盈亏（与初始资金比较）
        initial = self.history['initial_capital']
        snapshot['total_pnl'] = snapshot['total_assets'] - initial
        snapshot['total_pnl_pct'] = (snapshot['total_pnl'] / initial) * 100
        
        return snapshot
    
    def display_snapshot(self, snapshot):
        """显示账户快照"""
        print("\n" + "="*60)
        print(f"账户资产报告 - {snapshot['date']}")
        print("="*60)
        
        print(f"\n【资产概况】")
        print(f"  总资产: ${snapshot['total_assets']:,.2f}")
        print(f"  现金: ${snapshot['cash']:,.2f}")
        print(f"  持仓市值: ${snapshot['market_value']:,.2f}")
        
        print(f"\n【收益情况】")
        print(f"  今日盈亏: ${snapshot['daily_pnl']:,.2f} ({snapshot['daily_pnl_pct']:+.2f}%)")
        print(f"  总盈亏: ${snapshot['total_pnl']:,.2f} ({snapshot['total_pnl_pct']:+.2f}%)")
        
        if snapshot['positions']:
            print(f"\n【持仓明细】")
            for pos in snapshot['positions']:
                print(f"\n  {pos['symbol']}:")
                print(f"    数量: {pos['qty']} 股")
                print(f"    成本: ${pos['cost_price']:.2f}")
                print(f"    现价: ${pos['current_price']:.2f}")
                print(f"    市值: ${pos['market_value']:,.2f}")
                if pos['pnl'] != 0:
                    print(f"    盈亏: ${pos['pnl']:,.2f} ({pos['pnl_pct']:+.2%})")
        else:
            print(f"\n【持仓明细】")
            print("  暂无持仓")
        
        print("\n" + "="*60)
    
    def display_history_summary(self):
        """显示历史统计"""
        if not self.history['records']:
            print("暂无历史记录")
            return
        
        df = pd.DataFrame(self.history['records'])
        df['date'] = pd.to_datetime(df['date'])
        
        print("\n【历史统计】")
        print(f"  记录天数: {len(df)}")
        print(f"  初始资金: ${self.history['initial_capital']:,.2f}")
        print(f"  最高资产: ${df['total_assets'].max():,.2f} ({df.loc[df['total_assets'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
        print(f"  最低资产: ${df['total_assets'].min():,.2f} ({df.loc[df['total_assets'].idxmin(), 'date'].strftime('%Y-%m-%d')})")
        
        # 计算最大回撤
        cummax = df['total_assets'].cummax()
        drawdown = (df['total_assets'] - cummax) / cummax * 100
        max_dd = drawdown.min()
        print(f"  最大回撤: {max_dd:.2f}%")
        
        # 计算胜率（盈利天数）
        winning_days = len(df[df['daily_pnl'] > 0])
        losing_days = len(df[df['daily_pnl'] < 0])
        if winning_days + losing_days > 0:
            win_rate = winning_days / (winning_days + losing_days) * 100
            print(f"  日胜率: {win_rate:.1f}% ({winning_days}赢/{losing_days}输)")
        
        # 显示最近5天
        print(f"\n【最近5天】")
        recent = df.tail(5)
        for _, row in recent.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['total_assets']:,.2f} ({row['daily_pnl_pct']:+.2f}%)")
    
    def run(self):
        """主函数"""
        if not self.connect():
            return
        
        try:
            # 获取今日快照
            snapshot = self.get_account_snapshot()
            
            # 检查是否已有今日记录
            today = snapshot['date']
            existing_idx = None
            for i, record in enumerate(self.history['records']):
                if record['date'] == today:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # 更新今日记录
                self.history['records'][existing_idx] = snapshot
                print(f"已更新今日({today})的记录")
            else:
                # 添加新记录
                self.history['records'].append(snapshot)
                print(f"已添加今日({today})的记录")
            
            # 显示快照
            self.display_snapshot(snapshot)
            
            # 显示历史统计
            self.display_history_summary()
            
            # 保存记录
            self.save_history()
            
        finally:
            self.disconnect()


def main():
    tracker = AccountTracker()
    tracker.run()


if __name__ == "__main__":
    main()
