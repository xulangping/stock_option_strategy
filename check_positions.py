#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查当前持仓
"""

from futu import *

# 配置
US_SIM_ACC_ID = 16428245
UNLOCK_PWD = '153811'

def main():
    # 连接
    trd_ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.US,
        host='127.0.0.1', 
        port=11111, 
        security_firm=SecurityFirm.FUTUSECURITIES
    )
    
    # 解锁
    ret, data = trd_ctx.unlock_trade(UNLOCK_PWD)
    if ret != RET_OK:
        print(f"解锁失败: {data}")
        return
    
    # 查询持仓
    ret, positions = trd_ctx.position_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    print("\n" + "="*60)
    print("当前持仓状况")
    print("="*60)
    
    if ret == RET_OK:
        if positions.empty:
            print("\n✅ 账户已清空，没有任何持仓！")
        else:
            print(f"\n⚠️ 还有 {len(positions)} 个持仓:")
            for _, pos in positions.iterrows():
                print(f"\n  股票: {pos['code']}")
                print(f"  数量: {int(pos['qty'])} 股")
                print(f"  成本: ${pos['cost_price']:.2f}")
                print(f"  市值: ${float(pos['market_val']):,.2f}")
    
    # 查询账户信息
    ret, acc_info = trd_ctx.accinfo_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    print("\n" + "="*60)
    print("账户资金状况")
    print("="*60)
    
    if ret == RET_OK and not acc_info.empty:
        info = acc_info.iloc[0]
        print(f"\n总资产: ${float(info['total_assets']):,.2f}")
        print(f"现金: ${float(info['cash']):,.2f}")
        print(f"持仓市值: ${float(info['market_val']):,.2f}")
        print(f"可用资金: ${float(info['power']):,.2f}")
    
    print("\n" + "="*60)
    
    # 关闭连接
    trd_ctx.close()

if __name__ == "__main__":
    main()
