import os
import pandas as pd
import glob
from pathlib import Path

def read_all_parquet_files(dir_path):
    """
    读取指定文件夹中的所有parquet文件并整合成一个DataFrame
    
    Args:
        dir_path (str): 文件夹路径
        
    Returns:
        pd.DataFrame: 整合后的DataFrame
    """
    # 检查文件夹是否存在
    if not os.path.exists(dir_path):
        print(f"文件夹不存在: {dir_path}")
        return pd.DataFrame()
    
    # 获取所有parquet文件
    parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))
    
    if not parquet_files:
        print(f"在 {dir_path} 中没有找到parquet文件")
        return pd.DataFrame()
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 存储所有DataFrame
    dataframes = []
    
    # 逐个读取parquet文件
    for i, file_path in enumerate(parquet_files):
        try:
            print(f"正在读取文件 {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
            df = pd.read_parquet(file_path)
            
            # 添加文件来源信息
            df['source_file'] = os.path.basename(file_path)
            
            dataframes.append(df)
            print(f"  - 行数: {len(df)}, 列数: {len(df.columns)}")
            
        except Exception as e:
            print(f"  - 读取失败: {e}")
            continue
    
    if not dataframes:
        print("没有成功读取任何文件")
        return pd.DataFrame()
    
    # 合并所有DataFrame
    print(f"\n正在合并 {len(dataframes)} 个DataFrame...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"合并完成!")
    print(f"总行数: {len(combined_df)}")
    print(f"总列数: {len(combined_df.columns)}")
    print(f"列名: {list(combined_df.columns)}")
    
    return combined_df

def get_x_day_average_close_from_combined(symbol, date, X, combined_df):
    """
    从已合并的数据中计算X日均值（不包含指定日期）
    
    Args:
        symbol (str): 股票代码
        date (str): 日期，格式为 'YYYY-MM-DD'
        X (int): 天数
        combined_df (pd.DataFrame): 已合并的股票数据
        
    Returns:
        float: X日均值
    """
    try:
        # 转换日期格式
        target_date = pd.to_datetime(date)
        
        # 筛选出指定symbol的数据
        symbol_data = combined_df[combined_df['symbol'] == symbol].copy()
        # print(symbol_data.head(10))
        if symbol_data.empty:
            print(f"未找到 {symbol} 的数据")
            return None
        
        # 确保datetime列是datetime类型
        if 'date' in symbol_data.columns:
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            
            # 筛选出目标日期之前的数据
            before_date_data = symbol_data[symbol_data['date'] < target_date]

            if before_date_data.empty:
                print(f"未找到 {symbol} 在 {date} 之前的数据")
                return None
            
            # 按日期排序
            before_date_data = before_date_data.sort_values('date')
            
            # 获取最近的X天数据
            recent_data = before_date_data.tail(X)
            
            if len(recent_data) < X:
                print(f"数据不足：只有 {len(recent_data)} 天数据，需要 {X} 天")
                return None
            
            # 计算close价格的均值
            if 'close' in recent_data.columns:
                avg_close = recent_data['close'].mean()
                print(f"{symbol} 在 {date} 之前的 {X} 日均值: {avg_close:.2f}")
                return avg_close
            else:
                print(f"数据中没有 'close' 列")
                return None
        else:
            print("数据中没有 'datetime' 列")
            return None
            
    except Exception as e:
        print(f"计算 {symbol} 的 {X} 日均值时出错: {e}")
        return None


def MA_Bullish_Signal(symbol, date, combined_df):
    ma5 = get_x_day_average_close_from_combined(symbol = symbol, date = date, X = 5, combined_df = combined_df)
    ma10 = get_x_day_average_close_from_combined(symbol = symbol, date = date, X = 10, combined_df = combined_df)
    ma20 = get_x_day_average_close_from_combined(symbol = symbol, date = date, X = 20, combined_df = combined_df)
    if ma5 is None or ma10 is None or ma20 is None:
        return False    
    return ma5 > ma10 > ma20

class DynamicStopLoss:
    """动态止损策略类"""
    
    def __init__(self, initial_stop_loss: float = 0.05):
        """
        初始化动态止损策略
        
        参数:
        initial_stop_loss: 初始止损比例，默认5%
        """
        self.initial_stop_loss = initial_stop_loss
        self.stop_loss_levels = {}
        
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                              highest_price: float, trail_percent: float = 0.03) -> float:
        """
        移动止损/跟踪止损
        
        参数:
        entry_price: 入场价格
        current_price: 当前价格
        highest_price: 入场后的最高价
        trail_percent: 跟踪百分比
        
        返回:
        止损价格
        """
        if highest_price > entry_price:
            return highest_price * (1 - trail_percent)
        else:
            return entry_price * (1 - self.initial_stop_loss)


if __name__ == "__main__":
    dir_path = r"daily/"
    combined_df = read_all_parquet_files(dir_path)
    print(MA_Bullish_Signal(symbol = "A", date = "2024-10-22", combined_df = combined_df))