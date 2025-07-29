#!/usr/bin/env python3
"""
Historical Data Fetcher - 获取Unusual Whales的历史flow alerts数据
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from pathlib import Path

class HistoricalDataFetcher:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        self.api_key = os.getenv('UNUSUAL_WHALES_API_KEY')
        if not self.api_key:
            raise ValueError("请在.env文件中设置UNUSUAL_WHALES_API_KEY")
        
        self.base_url = "https://api.unusualwhales.com/api"
        self.headers = {
            'Accept': 'application/json, text/plain',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # 创建数据存储目录
        self.data_dir = Path("alert_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('historical_data_fetcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_flow_alerts_by_date_range(self, start_date, end_date, limit=200):
        """获取指定日期范围的flow alerts数据"""
        url = f"{self.base_url}/option-trades/flow-alerts"
        
        # 转换日期为ISO格式
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 看涨期权参数
        call_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_call': 'true',
            'is_otm': 'true',
            'max_diff': '1',           
            'min_diff': '0.01',        
            'max_dte': '200',           
            'min_premium': '100000',   
            'min_volume_oi_ratio': '1', 
            'issue_types[]': ['Common Stock', 'ETF', 'Index', 'ADR'],
            'newer_than': start_date.strftime('%Y-%m-%d'),
            'older_than': end_date.strftime('%Y-%m-%d'),
            'limit': str(limit)
        }
        
        # 看跌期权参数
        put_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_put': 'true',
            'is_otm': 'true',
            'max_diff': '1',           
            'min_diff': '0.01',        
            'max_dte': '200',           
            'min_premium': '100000',   
            'min_volume_oi_ratio': '1', 
            'issue_types[]': ['Common Stock', 'ETF', 'Index', 'ADR'],
            'newer_than': start_date.strftime('%Y-%m-%d'),
            'older_than': end_date.strftime('%Y-%m-%d'),
            'limit': str(limit)
        }
        
        all_alerts = []
        
        try:
            self.logger.info(f"获取历史数据，日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
            # 获取看涨期权数据
            self.logger.info("获取看涨期权历史数据...")
            call_response = requests.get(url, headers=self.headers, params=call_params)
            if call_response.status_code == 200:
                call_data = call_response.json()
                call_alerts = call_data.get('data', [])
                # 给每个call alert添加标识
                for alert in call_alerts:
                    alert['name'] = 'Flow alerts for All'
                all_alerts.extend(call_alerts)
                self.logger.info(f"获取到{len(call_alerts)}个看涨期权历史alerts")
            else:
                self.logger.error(f"获取看涨期权历史alerts失败: {call_response.status_code} - {call_response.text}")
            
            # 短暂延迟避免API限制
            time.sleep(1)
            
            # 获取看跌期权数据
            self.logger.info("获取看跌期权历史数据...")
            put_response = requests.get(url, headers=self.headers, params=put_params)
            if put_response.status_code == 200:
                put_data = put_response.json()
                put_alerts = put_data.get('data', [])
                # 给每个put alert添加标识
                for alert in put_alerts:
                    alert['name'] = 'Flow alerts for All put'
                all_alerts.extend(put_alerts)
                self.logger.info(f"获取到{len(put_alerts)}个看跌期权历史alerts")
            else:
                self.logger.error(f"获取看跌期权历史alerts失败: {put_response.status_code} - {put_response.text}")
            
            self.logger.info(f"总共获取到{len(all_alerts)}个历史flow alerts")
            return all_alerts
                
        except Exception as e:
            self.logger.error(f"获取历史flow alerts异常: {e}")
            return []
    
    def convert_to_backtest_format(self, api_alerts):
        """将API数据转换为回测需要的格式"""
        converted_alerts = []
        
        for alert in api_alerts:
            try:
                # 使用created_at作为执行时间
                executed_at = alert.get('created_at', datetime.now().isoformat() + 'Z')
                
                # 构建回测格式的alert
                converted_alert = {
                    'id': alert.get('id', ''),
                    'created_at': executed_at,
                    'name': alert.get('name', 'Flow alerts for All'),
                    'noti_type': 'flow_alerts',
                    'symbol': alert.get('option_chain', ''),
                    'symbol_type': 'option',
                    'tape_time': executed_at,
                    'user_noti_config_id': 'api_generated',
                    'meta': {
                        'executed_at': executed_at,
                        'underlying_symbol': alert.get('ticker', ''),
                        'total_premium': str(alert.get('total_premium', '0')),
                        'volume': alert.get('volume', 0),
                        'open_interest': alert.get('open_interest', 0),
                        'vol_oi_ratio': str(alert.get('volume_oi_ratio', '0')),
                        'strike_price': str(alert.get('strike', '0')),
                        'expiry': alert.get('expiry', ''),
                        'option_type': alert.get('type', 'call'),
                        'underlying_price': str(alert.get('underlying_price', '0')),
                        'price': str(alert.get('price', '0')),
                        'bid': str(alert.get('bid', '0')),
                        'ask': str(alert.get('ask', '0')),
                        # 历史数据可能没有这些字段，所以设为默认值
                        'bid_volume': alert.get('bid_volume', 0),
                        'ask_volume': alert.get('ask_volume', 0)
                    }
                }
                
                converted_alerts.append(converted_alert)
                
            except Exception as e:
                self.logger.error(f"转换alert数据失败: {e}, alert: {alert.get('id', 'unknown')}")
                continue
        
        self.logger.info(f"成功转换{len(converted_alerts)}个历史alert")
        return converted_alerts
    
    def save_alerts_to_json(self, alerts):
        """保存alert数据到JSON文件，按日期分组"""
        if not alerts:
            return
        
        # 按日期分组
        alerts_by_date = {}
        for alert in alerts:
            try:
                # 从created_at提取日期
                created_at = alert.get('created_at', '')
                if created_at:
                    date_str = created_at[:10]  # 取YYYY-MM-DD部分
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                if date_str not in alerts_by_date:
                    alerts_by_date[date_str] = []
                alerts_by_date[date_str].append(alert)
            except Exception as e:
                self.logger.error(f"处理alert日期失败: {e}")
                continue
        
        # 保存每个日期的alerts
        for date_str, daily_alerts in alerts_by_date.items():
            json_file = self.data_dir / f"alerts_{date_str}.json"
            
            # 加载现有数据
            existing_data = []
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    self.logger.error(f"加载现有JSON数据失败: {e}")
            
            # 合并数据，去重
            existing_ids = {alert.get('id') for alert in existing_data}
            new_alerts = [alert for alert in daily_alerts if alert.get('id') not in existing_ids]
            
            if new_alerts:
                existing_data.extend(new_alerts)
                
                # 保存到文件
                try:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"保存{len(new_alerts)}个新的历史alert到 {json_file}")
                except Exception as e:
                    self.logger.error(f"保存JSON文件失败: {e}")
            else:
                self.logger.info(f"日期{date_str}没有新的历史alert数据")
    
    def fetch_historical_data_by_day(self, start_date, end_date):
        """按天获取历史数据，避免单次请求数据过多"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        
        while current_date <= end_dt:
            next_date = current_date + timedelta(days=1)
            
            self.logger.info(f"获取日期 {current_date.strftime('%Y-%m-%d')} 的历史数据...")
            
            # 获取当天数据
            alerts = self.get_flow_alerts_by_date_range(current_date, next_date, limit=500)
            
            if alerts:
                # 转换格式并保存
                converted_alerts = self.convert_to_backtest_format(alerts)
                self.save_alerts_to_json(converted_alerts)
            else:
                self.logger.info(f"日期 {current_date.strftime('%Y-%m-%d')} 没有数据")
            
            # 延迟避免API限制
            time.sleep(2)
            
            current_date = next_date
        
        self.logger.info(f"历史数据获取完成，日期范围: {start_date} 到 {end_date}")
    
    def fetch_date_range(self, start_date, end_date):
        """获取指定日期范围的历史数据"""
        self.logger.info(f"开始获取历史数据，日期范围: {start_date} 到 {end_date}")
        
        try:
            # 按天获取数据
            self.fetch_historical_data_by_day(start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"获取历史数据异常: {e}")

def main():
    """主函数"""
    fetcher = HistoricalDataFetcher()
    
    import sys
    
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]  # 格式: 2025-07-16
        end_date = sys.argv[2]    # 格式: 2025-07-21
        fetcher.fetch_date_range(start_date, end_date)
    else:
        print("使用方法: python historical_data_fetcher.py <start_date> <end_date>")
        print("例如: python historical_data_fetcher.py 2025-07-16 2025-07-21")
        print("这将获取2025-07-16到2025-07-21的历史数据")

if __name__ == "__main__":
    main() 