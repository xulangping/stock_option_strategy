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
    
    def get_alert_configurations(self):
        """获取所有alert配置"""
        url = f"{self.base_url}/alerts/configuration"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"获取到{len(data.get('data', []))}个alert配置")
                return data.get('data', [])
            else:
                self.logger.error(f"获取alert配置失败: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"获取alert配置异常: {e}")
            return []
    
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
        self.logger.info("开始执行alert检查...")
        timestamp = datetime.now()
        
        # 1. 获取alert配置
        configs = self.get_alert_configurations()
        if not configs:
            self.logger.warning("未获取到alert配置")
            return
        
        # 2. 提取活跃配置的ID
        active_config_ids = [
            config['id'] for config in configs 
            if config.get('status') == 'active'
        ]
        
        if not active_config_ids:
            self.logger.warning("未找到活跃的alert配置")
            return
        
        self.logger.info(f"找到{len(active_config_ids)}个活跃的alert配置")
        
        # 3. 获取alert数据
        alerts = self.get_alerts_data(active_config_ids)
        if not alerts:
            self.logger.info("未获取到alert数据")
            return
        
        # 4. 保存数据
        self.save_alerts_to_json(alerts, timestamp)
        self.save_alerts_to_csv(alerts, timestamp)
        self.save_summary_report(alerts, timestamp)
        
        # 5. 保存已处理的alert ID
        self.save_processed_alerts()
        
        self.logger.info(f"alert检查完成，处理了{len(alerts)}个alert")
    
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