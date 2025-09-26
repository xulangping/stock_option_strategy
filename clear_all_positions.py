#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清空所有持仓 - 一键清仓脚本
"""

from futu import *
import time

# 配置
US_SIM_ACC_ID = 16428245
UNLOCK_PWD = '153811'

def main():
    print("="*60)
    print("清仓脚本 - 卖出所有持仓")
    print("="*60)
    
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
    print("✓ 交易解锁成功")
    
    # 获取当前持仓
    print("\n查询当前持仓...")
    ret, positions = trd_ctx.position_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    if ret != RET_OK:
        print(f"查询持仓失败: {positions}")
        return
    
    if positions.empty:
        print("✓ 没有持仓需要清空")
        trd_ctx.close()
        return
    
    print(f"\n发现 {len(positions)} 个持仓:")
    for _, pos in positions.iterrows():
        print(f"  {pos['code']}: {int(pos['qty'])} 股 @ ${pos['cost_price']:.2f}")
    
    # 自动确认清仓
    print("\n" + "="*60)
    print("自动确认清仓...")
    # confirm = input("确认清仓所有持仓？(输入 yes 确认): ")
    # if confirm.lower() != 'yes':
    #     print("取消清仓操作")
    #     trd_ctx.close()
    #     return
    
    # 逐个清仓
    print("\n开始清仓...")
    for _, pos in positions.iterrows():
        symbol = pos['code']
        qty = int(pos['qty'])
        
        print(f"\n卖出 {symbol}: {qty} 股...")
        
        # 使用市价单卖出
        ret, order_data = trd_ctx.place_order(
            price=0,  # 市价单
            qty=qty,
            code=symbol,
            trd_side=TrdSide.SELL,
            order_type=OrderType.MARKET,
            trd_env=TrdEnv.SIMULATE,
            acc_id=US_SIM_ACC_ID,
            session=Session.RTH  # 正常交易时段
        )
        
        if ret == RET_OK:
            order_id = order_data['order_id'][0]
            print(f"  ✓ 卖单提交成功，订单号: {order_id}")
            
            # 等待成交
            time.sleep(2)
            
            # 查询订单状态
            ret2, status = trd_ctx.order_list_query(
                order_id=order_id,
                trd_env=TrdEnv.SIMULATE,
                acc_id=US_SIM_ACC_ID
            )
            
            if ret2 == RET_OK and not status.empty:
                s = status.iloc[0]
                if s['order_status'] == 'FILLED_ALL':
                    print(f"  ✓ 成交: {s['dealt_qty']} 股 @ ${s['dealt_avg_price']:.2f}")
                else:
                    print(f"  状态: {s['order_status']}")
        else:
            print(f"  ✗ 下单失败: {order_data}")
    
    # 等待所有订单处理
    print("\n等待3秒确认所有订单...")
    time.sleep(3)
    
    # 再次查询持仓确认
    print("\n验证清仓结果...")
    ret, final_positions = trd_ctx.position_list_query(
        trd_env=TrdEnv.SIMULATE,
        acc_id=US_SIM_ACC_ID
    )
    
    if ret == RET_OK:
        if final_positions.empty:
            print("✓ 清仓成功！所有持仓已清空")
        else:
            print(f"⚠ 还有 {len(final_positions)} 个持仓未清空:")
            for _, pos in final_positions.iterrows():
                print(f"  {pos['code']}: {int(pos['qty'])} 股")
    
    # 显示账户信息
    print("\n" + "="*60)
    print("账户信息:")
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
    print("清仓操作完成！")

if __name__ == "__main__":
    main()
