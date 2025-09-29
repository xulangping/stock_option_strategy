#!/usr/bin/env python3
"""
集成网页文本监控交易系统
整合文本监控和富途交易功能
"""

import os
import json
import time
import logging
import warnings
from datetime import datetime
import pandas as pd

# 导入自定义模块
from web_text_parser import find_latest_text_file, parse_unusual_whales_text, extract_ticker_premium_only
from text_compare import has_significant_change, quick_content_check
from futu_strategy_executor import FutuStrategyExecutor

warnings.filterwarnings('ignore')

# 配置参数
WEB_TEXT_DIR = "/Users/zhanglijia/Documents/WebText"
DIFFERENCE_THRESHOLD = 0.001
JSON_STORAGE_DIR = "json_history"
EXTRACTED_JSON_DIR = "extracted_jsons"
CURRENT_TRADES_DIR = "current_trades"
BASELINE_DIR = "baseline_data"
CHECK_INTERVAL = 20

def get_current_json_filename():
    """获取当前日期的JSON文件名"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(CURRENT_TRADES_DIR, f"current_trades_{today}.json")

def get_baseline_json_filename():
    """获取基线数据的JSON文件名"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(BASELINE_DIR, f"baseline_{today}.json")

PROCESSED_FILES_FILE = "processed_files.json"

# 配置日志
def setup_logging():
    """设置日志配置，按日期保存"""
    today = datetime.now().strftime("%Y%m%d")
    log_filename = f"logs/integrated_web_trader_{today}.log"
    
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)
    
    # 清除现有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件handler - 启用立即刷新
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.flush = lambda: file_handler.stream.flush()  # 强制刷新
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    # 获取logger并设置自动刷新
    logger = logging.getLogger(__name__)
    
    # 重写logger方法以自动刷新
    original_info = logger.info
    original_error = logger.error
    original_warning = logger.warning
    
    def info_with_flush(*args, **kwargs):
        original_info(*args, **kwargs)
        file_handler.flush()
    
    def error_with_flush(*args, **kwargs):
        original_error(*args, **kwargs)
        file_handler.flush()
    
    def warning_with_flush(*args, **kwargs):
        original_warning(*args, **kwargs)
        file_handler.flush()
    
    logger.info = info_with_flush
    logger.error = error_with_flush
    logger.warning = warning_with_flush
    
    return logger

logger = setup_logging()

class IntegratedWebTrader:
    def __init__(self, initial_capital: float = 100000):
        """初始化集成网页文本监控交易系统"""
        # 初始化FutuStrategyExecutor
        self.futu_executor = FutuStrategyExecutor(initial_capital=initial_capital)
        
        # 文件处理状态
        self.processed_files = self.load_processed_files()
        
        # 创建必要的目录
        os.makedirs(JSON_STORAGE_DIR, exist_ok=True)
        os.makedirs(EXTRACTED_JSON_DIR, exist_ok=True)
        os.makedirs(CURRENT_TRADES_DIR, exist_ok=True)
        os.makedirs(BASELINE_DIR, exist_ok=True)
    
    def load_processed_files(self):
        """加载已处理的文件列表"""
        if os.path.exists(PROCESSED_FILES_FILE):
            with open(PROCESSED_FILES_FILE, 'r') as f:
                return json.load(f)
        return {"processed": [], "last_file": None, "last_hash": None}
    
    def save_processed_files(self):
        """保存已处理的文件列表"""
        with open(PROCESSED_FILES_FILE, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def get_unprocessed_files(self):
        """获取未处理的文件列表"""
        all_files = [f for f in os.listdir(WEB_TEXT_DIR) if f.endswith('.txt')]
        processed_set = set(self.processed_files["processed"])
        unprocessed = [f for f in all_files if f not in processed_set]
        unprocessed.sort(key=lambda x: os.path.getmtime(os.path.join(WEB_TEXT_DIR, x)))
        return unprocessed
    
    def compare_json_data(self, baseline_data, new_data):
        """比较数据，找出新增交易"""
        baseline_trades = {f"{t['ticker']}_{t['premium']}" for t in baseline_data["trades"]}
        new_trades = {f"{t['ticker']}_{t['premium']}" for t in new_data["trades"]}
        added_keys = new_trades - baseline_trades
        
        new_trade_items = []
        for trade in new_data["trades"]:
            if f"{trade['ticker']}_{trade['premium']}" in added_keys:
                new_trade_items.append(trade)
        
        return {"new_trades": new_trade_items}
    
    def create_trades_csv(self, new_trades_data, timestamp):
        """创建交易CSV文件"""
        trades_list = []
        for trade in new_trades_data["new_trades"]:
            trades_list.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '15:30:00',
                'underlying_symbol': trade['ticker'],
                'premium': trade['premium'],
                'underlying_price': 100.0
            })
        
        df = pd.DataFrame(trades_list)
        csv_path = os.path.join("real_time_option", f"trades_{timestamp}.csv")
        os.makedirs("real_time_option", exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        print(f"💾 交易CSV文件已创建: {csv_path}")
        logger.info(f"交易CSV文件已创建: {csv_path}")
        return csv_path
    
    def execute_new_trades(self, new_trades_data, timestamp):
        """执行新增交易并记录结果"""
        new_trades = new_trades_data["new_trades"]
        print(f"\n📈 开始执行 {len(new_trades)} 笔新增交易")
        logger.info(f"开始执行 {len(new_trades)} 笔新增交易")
        
        self.create_trades_csv(new_trades_data, timestamp)
        
        print("📊 新增交易详情:")
        for trade in new_trades:
            print(f"  📍 {trade['ticker']}: {trade['premium']:,}")
        
        execution_results = []
        
        for i, trade in enumerate(new_trades, 1):
            ticker = trade['ticker']
            premium = trade['premium']
            
            print(f"\n📋 [{i}/{len(new_trades)}] 处理交易: {ticker} - Premium: {premium:,}")
            logger.info(f"[{i}/{len(new_trades)}] 处理交易: {ticker} - Premium: {premium:,}")
            
            single_trade_df = pd.DataFrame([{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '15:30:00',
                'underlying_symbol': ticker,
                'premium': premium,
                'underlying_price': 100.0
            }])
            
            trade_result = {
                "ticker": ticker,
                "premium": premium,
                "timestamp": timestamp,
                "status": "unknown",
                "message": ""
            }
            
            old_orders_count = len([p for p in self.futu_executor.positions.values() if p.get('order_id') != 'PENDING'])
            
            self.futu_executor.process_option_signal(single_trade_df.iloc[0])
            new_orders_count = len([p for p in self.futu_executor.positions.values() if p.get('order_id') != 'PENDING'])
            
            if new_orders_count > old_orders_count:
                trade_result["status"] = "executed"
                trade_result["message"] = "交易已执行"
                print(f"✅ 交易已执行: {ticker}")
                logger.info(f"交易已执行: {ticker}")
            else:
                trade_result["status"] = "skipped"
                trade_result["message"] = "交易被跳过"
                print(f"⏭️  交易被跳过: {ticker}")
                logger.info(f"交易被跳过: {ticker}")
            
            execution_results.append(trade_result)
        
        self.update_current_trades_file(execution_results, timestamp)
    
    def update_current_trades_file(self, execution_results, timestamp):
        """更新当前交易文件"""
        current_json_file = get_current_json_filename()
        
        existing_trades = []
        if os.path.exists(current_json_file):
            with open(current_json_file, 'r', encoding='utf-8') as f:
                existing_trades = json.load(f)["trades"]
        
        existing_keys = {f"{t['ticker']}_{t['premium']}_{t['timestamp']}" for t in existing_trades}
        
        new_trades_added = 0
        for result in execution_results:
            trade_key = f"{result['ticker']}_{result['premium']}_{result['timestamp']}"
            if trade_key not in existing_keys:
                existing_trades.append(result)
                new_trades_added += 1
        
        updated_data = {
            "timestamp": timestamp,
            "last_update": datetime.now().isoformat(),
            "total_trades": len(existing_trades),
            "trades": existing_trades
        }
        
        with open(current_json_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 交易记录已更新: {current_json_file} (新增 {new_trades_added} 笔，累积 {len(existing_trades)} 笔)")
        logger.info(f"交易记录已更新: {current_json_file} (新增 {new_trades_added} 笔，累积 {len(existing_trades)} 笔)")
    
    def process_single_file(self, file_path):
        """处理单个文件"""
        filename = os.path.basename(file_path)
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n处理文件: {filename}")
        print(f"时间戳: {timestamp}")
        
        extracted_data = parse_unusual_whales_text(file_path)
        simplified_data = extract_ticker_premium_only(extracted_data)
        
        print(f"✅ 解析成功，找到 {len(simplified_data['trades'])} 条交易")
        
        extracted_json_path = os.path.join(EXTRACTED_JSON_DIR, f"extracted_{timestamp}.json")
        with open(extracted_json_path, 'w', encoding='utf-8') as f:
            json.dump({"timestamp": timestamp, "source_file": filename, "extracted_data": extracted_data}, 
                     f, indent=2, ensure_ascii=False)
        print(f"提取JSON已保存到: {extracted_json_path}")
        
        baseline_json_file = get_baseline_json_filename()
        
        if not os.path.exists(baseline_json_file):
            print("📝 这是第一个文件，创建基线数据...")
            
            with open(baseline_json_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            print(f"基线数据已保存: {baseline_json_file}")
            
            print(f"\n📊 基线交易列表:")
            for i, trade in enumerate(simplified_data['trades'], 1):
                print(f"  {i}. {trade['ticker']}: {trade['premium']:,}")
            
            print(f"\nℹ️  初始状态不执行交易，仅建立基线数据")
            print(f"\n✅ 基线已建立，包含 {len(simplified_data['trades'])} 条交易")
        else:
            latest_extracted = None
            extracted_files = [f for f in os.listdir(EXTRACTED_JSON_DIR) if f.startswith('extracted_') and f.endswith('.json')]
            extracted_files.sort()
            
            if len(extracted_files) >= 2:
                latest_extracted = os.path.join(EXTRACTED_JSON_DIR, extracted_files[-2])
            
            if latest_extracted:
                with open(latest_extracted, 'r', encoding='utf-8') as f:
                    previous_full_data = json.load(f)
                previous_simplified = extract_ticker_premium_only(previous_full_data["extracted_data"])
                print(f"📋 与上一次数据比较: {os.path.basename(latest_extracted)}")
            else:
                with open(baseline_json_file, 'r', encoding='utf-8') as f:
                    previous_simplified = json.load(f)
                print(f"📋 与基线数据比较")
            
            differences = self.compare_json_data(previous_simplified, simplified_data)
            
            if differences["new_trades"]:
                print(f"🔄 发现 {len(differences['new_trades'])} 笔新增交易")
                
                diff_json_path = os.path.join(JSON_STORAGE_DIR, f"difference_{timestamp}.json")
                with open(diff_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"timestamp": timestamp, **differences}, f, indent=2, ensure_ascii=False)
                print(f"差异JSON已保存到: {diff_json_path}")
                
                self.execute_new_trades(differences, timestamp)
            else:
                print("ℹ️  没有发现新增交易")
        
        self.processed_files["processed"].append(filename)
        self.processed_files["last_file"] = filename
        self.save_processed_files()
    
    def run(self):
        """运行监控系统"""
        logger.info("集成网页文本监控交易系统启动...")
        logger.info(f"监控目录: {WEB_TEXT_DIR}")
        logger.info(f"检查间隔: {CHECK_INTERVAL}秒")
        
        while True:
            unprocessed_files = self.get_unprocessed_files()
            
            if unprocessed_files:
                print(f"\n发现 {len(unprocessed_files)} 个未处理的文件")
                for file_name in unprocessed_files:
                    file_path = os.path.join(WEB_TEXT_DIR, file_name)
                    self.process_single_file(file_path)
            else:
                print(f"没有新文件，等待{CHECK_INTERVAL}秒...")
            
            time.sleep(CHECK_INTERVAL)

def main():
    """主入口"""
    trader = IntegratedWebTrader(initial_capital=100000)
    trader.run()

if __name__ == "__main__":
    main()