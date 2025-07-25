#!/usr/bin/env python3
"""
Alert Monitor - 每小时自动检查Unusual Whales的alert配置并保存数据
"""

import os
import json
import time
import requests
import pandas as pd
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
        
        # 存储已处理的alert ID，避免重复保存
        self.processed_alerts = set()
        self.load_processed_alerts()
    
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
    
    def load_processed_alerts(self):
        """加载已处理的alert ID"""
        processed_file = self.data_dir / "processed_alerts.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    self.processed_alerts = set(json.load(f))
                self.logger.info(f"已加载{len(self.processed_alerts)}个已处理的alert ID")
            except Exception as e:
                self.logger.error(f"加载已处理alert ID失败: {e}")
    
    def save_processed_alerts(self):
        """保存已处理的alert ID"""
        processed_file = self.data_dir / "processed_alerts.json"
        try:
            with open(processed_file, 'w') as f:
                json.dump(list(self.processed_alerts), f)
        except Exception as e:
            self.logger.error(f"保存已处理alert ID失败: {e}")
    
    def get_flow_alerts_from_api(self, newer_than_hours=24, limit=200):
        """从新的flow alerts API获取数据"""
        url = f"{self.base_url}/option-trades/flow-alerts"
        
        # 计算时间范围
        now = datetime.now()
        start_time = now - timedelta(hours=newer_than_hours)
        newer_than = start_time.strftime('%Y-%m-%d')
        
        # 获取看涨期权数据
        call_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_call': 'true',
            'is_otm': 'true',
            'max_diff': '1',           # Max % OTM: 1
            'min_diff': '0.01',        # Min % OTM: 0.01
            'max_dte': '60',           # Max DTE: 60
            'min_premium': '100000',   # Min Premium: 100000
            'min_volume_oi_ratio': '1', # Volume > OI (volume/oi > 1)
            'issue_types[]': ['Common Stock', 'ETF', 'Index', 'ADR'],
            'newer_than': newer_than,
            'limit': str(limit)
        }
        
        # 获取看跌期权数据
        put_params = {
            'all_opening': 'true',
            'is_ask_side': 'true', 
            'is_put': 'true',
            'is_otm': 'true',
            'max_diff': '1',           # Max % OTM: 1
            'min_diff': '0.01',        # Min % OTM: 0.01
            'max_dte': '60',           # Max DTE: 60
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
                all_alerts.extend(put_alerts)
                self.logger.info(f"获取到{len(put_alerts)}个看跌期权alerts")
            else:
                self.logger.error(f"获取看跌期权alerts失败: {put_response.status_code}")
            
            self.logger.info(f"总共获取到{len(all_alerts)}个flow alerts")
            return all_alerts
                
        except Exception as e:
            self.logger.error(f"获取flow alerts异常: {e}")
            return []
    
    def convert_api_data_to_original_format(self, api_alerts):
        """将新API数据转换为原始格式"""
        converted_alerts = []
        
        for alert in api_alerts:
            try:
                # 使用created_at作为执行时间（这是正确的UTC时间）
                created_at = alert.get('created_at', '')
                if created_at:
                    executed_at = created_at
                else:
                    executed_at = datetime.now().isoformat() + 'Z'
                
                # 根据期权类型设置名称
                option_type = alert.get('type', 'call')
                if option_type == 'put':
                    alert_name = 'Flow alerts for All put'
                else:
                    alert_name = 'Flow alerts for All'
                
                # 构建原始格式的alert
                converted_alert = {
                    'id': alert.get('id', ''),
                    'created_at': alert.get('created_at', executed_at),
                    'name': alert_name,  # 根据期权类型设置名称
                    'noti_type': 'flow_alerts',     # 固定类型
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
                        'iv': str(alert.get('iv_end', '0')),
                        'sector': alert.get('sector', ''),
                        'market_cap': str(alert.get('marketcap', '0')),
                        'alert_rule': alert.get('alert_rule', ''),
                        'total_size': alert.get('total_size', 0),
                        'trade_count': alert.get('trade_count', 0),
                        'has_floor': alert.get('has_floor', False),
                        'has_sweep': alert.get('has_sweep', False),
                        'price': str(alert.get('price', '0')),
                        'bid': str(alert.get('bid', '0')),
                        'ask': str(alert.get('ask', '0'))
                    }
                }
                
                converted_alerts.append(converted_alert)
                
            except Exception as e:
                self.logger.error(f"转换alert数据失败: {e}, alert: {alert.get('id', 'unknown')}")
                continue
        
        self.logger.info(f"成功转换{len(converted_alerts)}个alert到原始格式")
        return converted_alerts
    
    def get_alerts_data(self, config_ids, limit=200):
        """获取指定配置的alert数据 - 每个config_id单独请求"""
        url = f"{self.base_url}/alerts"
        all_alerts = []
        
        for config_id in config_ids:
            # 构建查询参数 - 每次只请求一个config_id
            params = {
                'limit': limit,
                'config_ids[]': [config_id]
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    alerts = data.get('data', [])
                    all_alerts.extend(alerts)
                    self.logger.info(f"配置ID {config_id} 获取到{len(alerts)}个alert数据")
                else:
                    self.logger.error(f"获取配置ID {config_id} 的alert数据失败: {response.status_code} - {response.text}")
                
                # 添加小延迟避免请求过于频繁
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"获取配置ID {config_id} 的alert数据异常: {e}")
                continue
        
        self.logger.info(f"总共获取到{len(all_alerts)}个alert数据")
        return all_alerts
    
    def save_alerts_to_json(self, alerts, timestamp):
        """保存alert数据到JSON文件"""
        if not alerts:
            return
        
        # 按日期分组保存
        date_str = timestamp.strftime('%Y-%m-%d')
        json_file = self.data_dir / f"alerts_{date_str}.json"
        
        # 加载现有数据
        existing_data = []
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                self.logger.error(f"加载现有JSON数据失败: {e}")
        
        # 添加新数据
        new_alerts = []
        for alert in alerts:
            if alert['id'] not in self.processed_alerts:
                new_alerts.append(alert)
                self.processed_alerts.add(alert['id'])
        
        if new_alerts:
            existing_data.extend(new_alerts)
            
            # 保存到文件
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"保存{len(new_alerts)}个新alert到 {json_file}")
            except Exception as e:
                self.logger.error(f"保存JSON文件失败: {e}")
    
    def save_alerts_to_csv(self, alerts, timestamp):
        """保存alert数据到CSV文件"""
        if not alerts:
            return
        
        # 过滤新的alert
        new_alerts = [alert for alert in alerts if alert['id'] not in self.processed_alerts]
        if not new_alerts:
            return
        
        # 按日期分组保存
        date_str = timestamp.strftime('%Y-%m-%d')
        csv_file = self.data_dir / f"alerts_{date_str}.csv"
        
        # 展平数据结构
        flattened_data = []
        for alert in new_alerts:
            flat_alert = {
                'id': alert['id'],
                'created_at': alert['created_at'],
                'name': alert['name'],
                'noti_type': alert['noti_type'],
                'symbol': alert['symbol'],
                'symbol_type': alert['symbol_type'],
                'tape_time': alert['tape_time'],
                'user_noti_config_id': alert['user_noti_config_id']
            }
            
            # 展平meta字段
            meta = alert.get('meta', {})
            for key, value in meta.items():
                flat_alert[f'meta_{key}'] = value
            
            flattened_data.append(flat_alert)
        
        # 转换为DataFrame
        df = pd.DataFrame(flattened_data)
        
        # 如果文件存在，追加数据
        if csv_file.exists():
            try:
                existing_df = pd.read_csv(csv_file)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['id'], keep='last')
            except Exception as e:
                self.logger.error(f"加载现有CSV数据失败: {e}")
        
        # 保存到文件
        try:
            df.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"保存{len(new_alerts)}个新alert到 {csv_file}")
        except Exception as e:
            self.logger.error(f"保存CSV文件失败: {e}")
    
    def save_summary_report(self, alerts, timestamp):
        """保存汇总报告"""
        if not alerts:
            return
        
        # 统计信息
        stats = {
            'timestamp': timestamp.isoformat(),
            'total_alerts': len(alerts),
            'by_noti_type': {},
            'by_symbol_type': {},
            'recent_alerts': []
        }
        
        # 按通知类型统计
        for alert in alerts:
            noti_type = alert.get('noti_type', 'unknown')
            stats['by_noti_type'][noti_type] = stats['by_noti_type'].get(noti_type, 0) + 1
        
        # 按符号类型统计
        for alert in alerts:
            symbol_type = alert.get('symbol_type', 'unknown')
            stats['by_symbol_type'][symbol_type] = stats['by_symbol_type'].get(symbol_type, 0) + 1
        
        # 最近的alert
        sorted_alerts = sorted(alerts, key=lambda x: x.get('created_at', ''), reverse=True)
        stats['recent_alerts'] = sorted_alerts[:10]
        
        # 保存汇总报告
        report_file = self.data_dir / "latest_summary.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"保存汇总报告到 {report_file}")
        except Exception as e:
            self.logger.error(f"保存汇总报告失败: {e}")
    
    def run_once(self):
        """执行一次检查"""
        self.logger.info("开始执行flow alerts检查...")
        timestamp = datetime.now()
        
        # 1. 从新的flow alerts API获取数据
        api_alerts = self.get_flow_alerts_from_api(newer_than_hours=24, limit=200)
        if not api_alerts:
            self.logger.info("未获取到flow alerts数据")
            return
        
        # 2. 转换数据格式为原始格式
        alerts = self.convert_api_data_to_original_format(api_alerts)
        if not alerts:
            self.logger.warning("数据转换后为空")
            return
        
        # 3. 保存数据（使用原来的保存逻辑）
        self.save_alerts_to_json(alerts, timestamp)
        self.save_alerts_to_csv(alerts, timestamp)
        self.save_summary_report(alerts, timestamp)
        
        # 4. 保存已处理的alert ID
        self.save_processed_alerts()
        
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
                time.sleep(300)  # 等待5分钟后重试

def main():
    """主函数"""
    monitor = AlertMonitor()
    
    # 可以选择运行一次或持续运行
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # 运行一次
        monitor.run_once()
    else:
        # 持续运行
        monitor.run_continuous(interval_hours=1)

if __name__ == "__main__":
    main() 