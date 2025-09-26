#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看富途交易记录和持仓
"""

from futu import *
import datetime

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
    trd_ctx.unlock_trade(UNLOCK_PWD)
    
    print("="*60)
    print("富途交易记录查询")
    print("="*60)
    
    # 1. 查询今天的订单
    print("\n【今日订单】")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    ret, orders = trd_ctx.history_order_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID,
        start=today,
        end=today
    )
    
    if ret == RET_OK and not orders.empty:
        print(f"今天共 {len(orders)} 个订单：")
        for idx, order in orders.iterrows():
            print(f"\n订单 #{idx+1}:")
            print(f"  订单号: {order['order_id']}")
            print(f"  股票: {order['code']}")
            print(f"  方向: {order['trd_side']}")
            print(f"  数量: {order['qty']}")
            print(f"  价格: ${order['price']:.2f}")
            print(f"  状态: {order['order_status']}")
            print(f"  已成交: {order['dealt_qty']} 股")
            if order['dealt_qty'] > 0:
                print(f"  成交均价: ${order['dealt_avg_price']:.2f}")
            print(f"  创建时间: {order['create_time']}")
    else:
        print("今天没有订单")
    
    # 2. 查询成交记录
    print("\n【成交记录】")
    ret, deals = trd_ctx.history_deal_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID,
        start=today,
        end=today
    )
    
    if ret == RET_OK and not deals.empty:
        print(f"今天共 {len(deals)} 笔成交：")
        for idx, deal in deals.iterrows():
            print(f"\n成交 #{idx+1}:")
            print(f"  订单号: {deal['order_id']}")
            print(f"  股票: {deal['code']}")
            print(f"  方向: {deal['trd_side']}")
            print(f"  成交数量: {deal['qty']}")
            print(f"  成交价格: ${deal['price']:.2f}")
            print(f"  成交时间: {deal['create_time']}")
    else:
        print("今天没有成交")
    
    # 3. 查询当前持仓
    print("\n【当前持仓】")
    ret, positions = trd_ctx.position_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    if ret == RET_OK and not positions.empty:
        print(f"共 {len(positions)} 个持仓：")
        for idx, pos in positions.iterrows():
            print(f"\n{pos['code']}:")
            print(f"  数量: {pos['qty']} 股")
            print(f"  成本价: ${pos['cost_price']:.2f}")
            print(f"  当前价: ${pos['nominal_price']:.2f}")
            print(f"  市值: ${pos['market_val']:.2f}")
            pl_val = pos.get('pl_val', 0)
            pl_ratio = pos.get('pl_ratio', 0)
            if pl_val != 0:
                print(f"  盈亏: ${pl_val:.2f} ({pl_ratio:.2%})")
    else:
        print("暂无持仓")
    
    # 4. 查询账户信息
    print("\n【账户信息】")
    ret, acc_info = trd_ctx.accinfo_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    if ret == RET_OK and not acc_info.empty:
        info = acc_info.iloc[0]
        print(f"  总资产: ${float(info['total_assets']):,.2f}")
        print(f"  现金: ${float(info['cash']):,.2f}")
        print(f"  持仓市值: ${float(info['market_val']):,.2f}")
    
    # 关闭连接
    trd_ctx.close()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
