# Stock Option Strategy - Alert Monitor

## 功能说明

这个项目包含一个自动化的期权alert监控系统，用于定期检查Unusual Whales的alert配置并保存数据。

## 主要功能

### Alert Monitor (`alert_monitor.py`)
- 每小时自动检查Unusual Whales的alert配置
- 获取符合条件的alert数据
- 多格式保存数据（JSON、CSV、汇总报告）
- 避免重复保存已处理的alert
- 完整的日志记录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

创建 `.env` 文件：

```
# Unusual Whales API Token
UNUSUAL_WHALES_API_KEY=your_api_key_here
```

## 使用方法

### 1. 运行一次检查
```bash
python alert_monitor.py --once
```

### 2. 持续监控（每小时检查一次）
```bash
python alert_monitor.py
```

### 3. 后台运行
```bash
nohup python alert_monitor.py > monitor.log 2>&1 &
```

## 数据存储

脚本会创建 `alert_data/` 目录，包含：

- `alerts_YYYY-MM-DD.json` - 每日的完整alert数据（JSON格式）
- `alerts_YYYY-MM-DD.csv` - 每日的alert数据（CSV格式，便于分析）
- `latest_summary.json` - 最新的汇总报告
- `processed_alerts.json` - 已处理的alert ID列表

## 日志记录

- 控制台输出实时日志
- `alert_monitor.log` 文件记录详细日志

## 数据格式

### Alert数据结构
```json
{
  "id": "alert_id",
  "created_at": "2024-12-11T14:00:00Z",
  "name": "Alert Name",
  "noti_type": "flow_alerts",
  "symbol": "STOCK_SYMBOL",
  "symbol_type": "stock",
  "tape_time": "2024-12-11T14:00:00Z",
  "user_noti_config_id": "config_id",
  "meta": {
    // 具体的alert元数据
  }
}
```

### 汇总报告结构
```json
{
  "timestamp": "2024-12-11T14:00:00",
  "total_alerts": 100,
  "by_noti_type": {
    "flow_alerts": 50,
    "dividends": 30,
    "trading_state": 20
  },
  "by_symbol_type": {
    "stock": 90,
    "option": 10
  },
  "recent_alerts": [
    // 最近10个alert
  ]
}
```

## 特性

- ✅ 自动去重，避免重复保存
- ✅ 多格式数据存储
- ✅ 完整的错误处理和日志记录
- ✅ 支持一次性运行和持续监控
- ✅ 数据按日期分组存储
- ✅ 汇总报告生成

## 注意事项

1. 确保 `.env` 文件中的API密钥有效
2. 脚本会自动创建必要的目录
3. 建议在稳定的网络环境下运行
4. 可以通过 `Ctrl+C` 优雅地停止持续监控 