# step1：当日涨幅 3-5
# step2：量比筛选，当日量比小于1剔除
# step3: 换手率筛选，保留当日换手率5-10%
# step4：市值筛选 50 - 200
# step5：成交量，保留成交量持续放大
# step6：个股K线， 短期5/10/20日均线，搭配60日均线多头向上发散
# step7：分时图跑赢大盘，配合当下热点板块，分时图需全天在大盘分时图价格上方
# step8：最后30min创出当日新高，可介入，个股回落均线，不跌破入场

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import shutil



def get_stock_vs_market_strength(stock_code, date, period="5", days_back=5):
    """
    判断个股分钟级K线是否强于大盘
    
    参数:
    stock_code: 股票代码 (如 "600036" 招商银行)
    period: 时间粒度 ("1", "5", "15", "30", "60")
    days_back: 回溯天数
    
    返回:
    strength_ratio: 个股强于大盘的时间点占比
    result_df: 包含比较结果的DataFrame
    """
    # 获取当前日期
    end_date = date
    start_date = date
    
    try:
        # 获取个股分钟级数据
        stock_data = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            period=period,
            start_date=f"{start_date} 09:30:00",
            end_date=f"{end_date} 15:00:00"
        )
        
        # 获取上证指数分钟级数据
        market_data = ak.index_zh_a_hist_min_em(
            symbol="000001",  # 上证指数
            period=period,
            start_date=f"{start_date} 09:30:00",
            end_date=f"{end_date} 15:00:00"
        )
        
        # 数据预处理
        stock_data = stock_data.copy()
        market_data = market_data.copy()
        
        # 确保时间列格式正确
        stock_data['时间'] = pd.to_datetime(stock_data['时间'])
        market_data['时间'] = pd.to_datetime(market_data['时间'])
        
        # 设置时间索引
        stock_data.set_index('时间', inplace=True)
        market_data.set_index('时间', inplace=True)
        
        # 计算涨跌幅
        stock_data['个股涨跌幅'] = stock_data['收盘'].pct_change() * 100
        market_data['大盘涨跌幅'] = market_data['收盘'].pct_change() * 100
        
        # 合并数据
        merged_data = pd.merge(
            stock_data[['收盘', '个股涨跌幅']], 
            market_data[['收盘', '大盘涨跌幅']], 
            left_index=True, 
            right_index=True, 
            how='inner',
            suffixes=('_个股', '_大盘')
        )
        
        # 计算相对强度
        merged_data['相对强度'] = merged_data['个股涨跌幅'] - merged_data['大盘涨跌幅']
        merged_data['强于大盘'] = merged_data['相对强度'] > 0
        
        # 计算统计指标
        total_points = len(merged_data)
        strong_points = merged_data['强于大盘'].sum()
        strength_ratio = strong_points / total_points if total_points > 0 else 0
        
        # 计算平均相对强度
        avg_strength = merged_data['相对强度'].mean()
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            '股票代码': [stock_code],
            '分析周期': [f"{period}分钟"],
            '分析时段': [f"{start_date} 至 {end_date}"],
            '总数据点': [total_points],
            '强于大盘点数': [strong_points],
            '强于大盘占比': [f"{strength_ratio:.2%}"],
            '平均相对强度': [f"{avg_strength:.4f}%"],
            '结论': ['强于大盘' if strength_ratio > 0.5 else '弱于大盘']
        })
        # print(result_df)
        return strength_ratio, merged_data, result_df
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return 0, pd.DataFrame(), pd.DataFrame()


def is_increasing_array(arr, strict=True):
    """
    判断数组是否为递增数组
    
    参数:
    arr: 输入数组
    strict: 是否要求严格递增（默认True）
           True - 严格递增：每个元素必须大于前一个元素
           False - 非严格递增：每个元素必须大于或等于前一个元素
    
    返回:
    bool: 如果是递增数组返回True，否则返回False
    """
    if not arr or len(arr) == 1:
        return True  # 空数组或只有一个元素的数组被认为是递增的
    
    if strict:
        # 检查是否严格递增
        for i in range(1, len(arr)):
            if arr[i] <= arr[i-1]:
                return False
    else:
        # 检查是否非严格递增
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                return False
    
    return True


def get_date(days_back):
    """获取指定天数前的日期"""
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")

def get_stock_hist_data(stock_code, today, days_back):
    hist_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=get_date(days_back), end_date=today)
    return hist_data

def save_hist_stock_data(data, today, days_back, data_dir = "./data/"):
    stock_code_list = data["代码"].tolist()
    process_nums = 0
    for stock_code in stock_code_list:
        try:
            if len(str(stock_code)) !=6 :
                pass
            else:
                df = get_stock_hist_data(str(stock_code), today, days_back)
                save_path = data_dir + "/" + str(today) + "_" + str(stock_code) + ".csv"
                df.to_csv(save_path)
                process_nums += 1
        except Exception:
            print("==== error stock code =====", stock_code)

def volume_increase(data, today, days = 3, data_dir = "./data/"):
    stock_code_list = data["代码"].tolist()
    return_list = []
    for stock_code in stock_code_list:
        if len(str(stock_code)) !=6 :
            pass
        else:
            save_path = data_dir + "/" + str(today) + "_" + str(stock_code) + ".csv"
            df = pd.read_csv(save_path)         
            if is_increasing_array(list(df["成交量"].tail(days).values)):
                return_list.append(stock_code)
    return return_list
         

def filter_kechuang(code_list):
    ret = []
    for code in code_list:
        if not str(code).startswith("688"):
            ret.append(code)
    return ret

def ma_trend(code_list, today, data_dir = "./data/"):
    ret_code = []
    for code in code_list:
        save_path = data_dir + "/" + str(today) + "_" + str(code) + ".csv"
        df = pd.read_csv(save_path)
        ma5 = df["收盘"].rolling(5).mean()
        ma10 = df["收盘"].rolling(10).mean()
        ma20 = df["收盘"].rolling(20).mean()
        ma30 = df["收盘"].rolling(30).mean()
        array_list = [ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1], ma30.iloc[-1]][::-1]
        if is_increasing_array(array_list):
            ret_code.append(code)
    return ret_code

def get_stock_data(today, data_dir = "./data/"):
    save_path = data_dir + "/" + str(today) + ".csv"
    df = ak.stock_zh_a_spot_em()
    df.to_csv(save_path)
    return df


strategy = {
    "涨跌幅": {"min": 3, "max": 11},
    "量比": 1,
    "换手率": {"min": 5, "max": 10},
    "市值": {"min": 50, "max": 200},
}

def basic_filter(df, strategy):
    step1_data = df[(df["涨跌幅"] >= strategy["涨跌幅"]["min"]) & (df["涨跌幅"] <= strategy["涨跌幅"]["max"])]
    step2_data = step1_data[step1_data["量比"] > strategy["量比"]]
    step3_data = step2_data[(step2_data["换手率"] > strategy["换手率"]["min"]) & (step2_data["换手率"] < strategy["换手率"]["max"])]
    step3_data["总市值"] = step3_data["总市值"] / 100000000
    step4_data = step3_data[(step3_data["总市值"] > strategy["市值"]["min"]) & (step3_data["总市值"] < strategy["市值"]["max"])]
    return step4_data

def get_previous_trading_day(date=None):
    """
    获取指定日期的上一个交易日
    
    参数:
    date: 日期字符串或datetime对象，默认为当前日期
    
    返回:
    previous_trading_day: 上一个交易日的日期字符串 (YYYY-MM-DD格式)
    """
    # 如果没有指定日期，使用当前日期
    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    
    # 获取最近一年的交易日历
    trade_dates = ak.tool_trade_date_hist_sina()
    
    # 确保日期格式正确
    trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date']).dt.date
    
    # 转换为日期列表
    trading_days = sorted(trade_dates['trade_date'].tolist(), reverse=True)
    
    # 将输入日期转换为日期对象
    input_date = date.date() if isinstance(date, datetime) else date
    
    # 找到输入日期之前的交易日
    for trading_day in trading_days:
        if trading_day < input_date:
            return trading_day.strftime('%Y-%m-%d')
    
    # 如果没有找到，返回None
    return None

if __name__ == "__main__":
    start = time.time()
    today = datetime.now().strftime("%Y%m%d")
    dir_path = "./data_" + str(today) + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    df = get_stock_data(today, data_dir = dir_path)
    df = basic_filter(df, strategy)
    save_hist_stock_data(df, today, days_back = 60, data_dir = dir_path)
    res_list = volume_increase(df, today, days = 3, data_dir = dir_path)
    res_list = filter_kechuang(res_list)    
    res_list = ma_trend(res_list , today, data_dir = dir_path)
    print(res_list)
    # res_list = ['002082', '000048', '002183', '300882', '000758', '002955', '605577']
    final_list = []
    trading_day = datetime.now().strftime("%Y-%m-%d")
    print("last tarding day is ", trading_day)
    for val in res_list:
        strength_ratio, merged_data, result_df = get_stock_vs_market_strength(str(val), trading_day, period="1", days_back=5)
        if strength_ratio > 0.5:
            final_list.append((val, strength_ratio))
    print(final_list)
    end = time.time()
    print("time cost is ", end - start)

    