#!/usr/bin/env python3
"""
Alert Monitor - 每小时自动检查Unusual Whales的flow alerts并保存数据
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from pathlib import Path

class AlertMonitor:
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
                logging.FileHandler('alert_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_flow_alerts_from_api(self, newer_than_hours=24, limit=200):
        """从flow alerts API获取数据，使用两套参数（call和put）"""
        url = f"{self.base_url}/option-trades/flow-alerts"
        
        # 计算时间范围
        now = datetime.now()
        start_time = now - timedelta(hours=newer_than_hours)
        newer_than = start_time.strftime('%Y-%m-%d')
        
        # 看涨期权参数
        call_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_call': 'true',
            'is_otm': 'true',
            'max_diff': '1',           # Max % OTM: 1
            'min_diff': '0.01',        # Min % OTM: 0.01
            'max_dte': '200',           # Max DTE: 60
            'min_premium': '100000',   # Min Premium: 100000
            'min_volume_oi_ratio': '1', # Volume > OI (volume/oi > 1)
            'issue_types[]': ['Common Stock', 'ETF', 'Index', 'ADR'],
            'newer_than': newer_than,
            'limit': str(limit)
        }
        
        # 看跌期权参数
        put_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_put': 'true',
            'is_otm': 'true',
            'max_diff': '1',           # Max % OTM: 1
            'min_diff': '0.01',        # Min % OTM: 0.01
            'max_dte': '200',           # Max DTE: 200
            'min_premium': '100000',   # Min Premium: 100000
            'min_volume_oi_ratio': '1', # Volume > OI (volume/oi > 1)
            'issue_types[]': ['Common Stock', 'ETF', 'Index', 'ADR'],
            'newer_than': newer_than,
            'limit': str(limit)
        }
        
        all_alerts = []
        
        try:
            self.logger.info(f"从flow alerts API获取数据，时间范围: {newer_than} 至今")
            
            # 获取看涨期权数据
            self.logger.info("获取看涨期权数据...")
            call_response = requests.get(url, headers=self.headers, params=call_params)
            if call_response.status_code == 200:
                call_data = call_response.json()
                call_alerts = call_data.get('data', [])
                # 给每个call alert添加标识
                for alert in call_alerts:
                    alert['name'] = 'Flow alerts for All'
                all_alerts.extend(call_alerts)
                self.logger.info(f"获取到{len(call_alerts)}个看涨期权alerts")
            else:
                self.logger.error(f"获取看涨期权alerts失败: {call_response.status_code}")
            
            # 获取看跌期权数据
            self.logger.info("获取看跌期权数据...")
            put_response = requests.get(url, headers=self.headers, params=put_params)
            if put_response.status_code == 200:
                put_data = put_response.json()
                put_alerts = put_data.get('data', [])
                # 给每个put alert添加标识
                for alert in put_alerts:
                    alert['name'] = 'Flow alerts for All put'
                all_alerts.extend(put_alerts)
                self.logger.info(f"获取到{len(put_alerts)}个看跌期权alerts")
            else:
                self.logger.error(f"获取看跌期权alerts失败: {put_response.status_code}")
            
            self.logger.info(f"总共获取到{len(all_alerts)}个flow alerts")
            return all_alerts
                
        except Exception as e:
            self.logger.error(f"获取flow alerts异常: {e}")
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
                    'name': alert.get('name', 'Flow alerts for All'),  # 使用之前添加的name
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
                        'ask': str(alert.get('ask', '0'))
                    }
                }
                
                converted_alerts.append(converted_alert)
                
            except Exception as e:
                self.logger.error(f"转换alert数据失败: {e}, alert: {alert.get('id', 'unknown')}")
                continue
        
        self.logger.info(f"成功转换{len(converted_alerts)}个alert")
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
                    self.logger.info(f"保存{len(new_alerts)}个新alert到 {json_file}")
                except Exception as e:
                    self.logger.error(f"保存JSON文件失败: {e}")
            else:
                self.logger.info(f"日期{date_str}没有新的alert数据")
    
    def run_once(self):
        """执行一次检查"""
        self.logger.info("开始执行flow alerts检查...")
        
        # 1. 从API获取数据
        api_alerts = self.get_flow_alerts_from_api(newer_than_hours=24*3, limit=200)
        if not api_alerts:
            self.logger.info("未获取到flow alerts数据")
            return
        
        # 2. 转换数据格式
        alerts = self.convert_to_backtest_format(api_alerts)
        if not alerts:
            self.logger.warning("数据转换后为空")
            return
        
        # 3. 保存数据
        self.save_alerts_to_json(alerts)
        
        self.logger.info(f"flow alerts检查完成，处理了{len(alerts)}个alert")
    
    def run_continuous(self, interval_hours=1):
        """持续运行，每隔指定小时检查一次"""
        self.logger.info(f"开始持续监控，每{interval_hours}小时检查一次...")
        
        while True:
            try:
                self.run_once()
                
                # 等待指定时间
                sleep_seconds = interval_hours * 3600
                self.logger.info(f"等待{interval_hours}小时后进行下次检查...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("用户中断，停止监控")
                break
            except Exception as e:
                self.logger.error(f"运行异常: {e}")
                self.logger.info("等待5分钟后重试...")
                time.sleep(300)

def main():
    """主函数"""
    monitor = AlertMonitor()
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        monitor.run_once()
    else:
        monitor.run_continuous(interval_hours=1)

if __name__ == "__main__":
    main() 