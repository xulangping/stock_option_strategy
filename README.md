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

### Backtest Strategy (`backtest_strategy.py`)
- 基于历史option flow alert数据进行回测
- 自动选择每日异常分数最高的交易机会
- 支持多种退出策略（次日开盘、当日收盘）
- 详细的交易记录和性能分析
- 风险控制：每日最多5笔交易，每笔20%仓位

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

### Alert监控系统

#### 1. 运行一次检查
```bash
python alert_monitor.py --once
```

#### 2. 持续监控（每小时检查一次）
```bash
python alert_monitor.py
```

#### 3. 后台运行
```bash
nohup python alert_monitor.py > monitor.log 2>&1 &
```

### 回测系统

#### 1. 运行回测策略
```bash
python backtest_strategy.py
```

#### 2. 获取价格数据（需要先运行）
```bash
python daily_price_fetcher.py
```

**注意**：运行回测前需要确保：
- `alert_data/` 目录包含alert数据文件（`alerts_YYYY-MM-DD.json`）
- `price_data/` 目录包含对应的5分钟K线数据（`SYMBOL_5min.csv`）

## 数据存储

### Alert数据
脚本会创建 `alert_data/` 目录，包含：

- `alerts_YYYY-MM-DD.json` - 每日的完整alert数据（JSON格式）
- `alerts_YYYY-MM-DD.csv` - 每日的alert数据（CSV格式，便于分析）
- `latest_summary.json` - 最新的汇总报告
- `processed_alerts.json` - 已处理的alert ID列表

### 价格数据
`price_data/` 目录包含：

- `SYMBOL_5min.csv` - 各股票的5分钟K线数据

### 回测结果
回测系统会生成：

- `backtest_results_v3.json` - 详细的回测结果（JSON格式）
- 控制台输出完整的交易记录和性能统计

## 日志记录

- 控制台输出实时日志
- `alert_monitor.log` 文件记录详细日志

## 数据格式

### 回测策略说明

#### 策略逻辑
1. **数据过滤**：只处理"Flow alerts for All"和"Flow alerts for All put"类型的alert
2. **去重处理**：基于执行时间、底层股票和期权代码进行去重
3. **异常评分**：根据交易premium和到期天数计算异常分数 (premium / days_to_expiration)
4. **每日选择**：每个交易日选择异常分数最高的前10个alert
5. **最终筛选**：如果当日交易超过5个，只选择异常分数最高的前5个
6. **仓位管理**：每笔交易20%仓位，每日最多100%仓位

#### 交易逻辑
- **进场时间**：alert执行时间后5分钟
- **进场价格**：使用股票开盘价
- **盘后处理**：如果是盘后信号，使用次日开盘价进场
- **退出策略**：
  - 优先：次日开盘价退出
  - 备选：当日收盘时退出（使用开盘价）
- **交易方向**：根据期权价格与bid/ask关系自动判断做多/做空

#### 输出结果
- **总体统计**：总交易次数、总收益率、胜率、平均收益等
- **交易明细**：每笔交易的详细信息（日期、股票、类型、方向、买入时间、价格、收益等）
- **退出原因统计**：各种退出原因的分布

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

### Alert监控系统
- ✅ 自动去重，避免重复保存
- ✅ 多格式数据存储
- ✅ 完整的错误处理和日志记录
- ✅ 支持一次性运行和持续监控
- ✅ 数据按日期分组存储
- ✅ 汇总报告生成

### 回测系统
- ✅ 智能异常评分算法，筛选高质量交易机会
- ✅ 自动去重，避免重复交易同一信号
- ✅ 灵活的退出策略（次日开盘优先）
- ✅ 自动判断交易方向（基于期权bid/ask分析）
- ✅ 严格的风险控制（每日最多5笔交易）
- ✅ 详细的性能分析和交易记录
- ✅ 支持盘后信号处理
- ✅ 统一使用开盘价进出场，减少数据偏差

## 注意事项

### Alert监控系统
1. 确保 `.env` 文件中的API密钥有效
2. 脚本会自动创建必要的目录
3. 建议在稳定的网络环境下运行
4. 可以通过 `Ctrl+C` 优雅地停止持续监控

### 回测系统
1. 运行回测前必须先获取对应的价格数据
2. Alert数据和价格数据的日期范围需要匹配
3. 回测结果仅供参考，实际交易存在滑点、手续费等成本
4. 系统假设所有交易都能成功执行，实际情况可能有差异
5. 建议在充足的历史数据基础上进行回测分析 