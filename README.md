# 🚀 astock-agent — A股智能分析 Agent 工具

> **使用最新实时数据进行选股、多维分析、策略回测与每日操作建议的完整工具链**
> 
> 支持 MCP Server（Agent 接口）+ CLI 双模式运行

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| **📊 六维深度分析** | 技术面 / 估值 / 资金面 / 基本面 / 财报质量 / 舆情情绪，每维度 0~100 分 |
| **🔍 智能选股** | 多条件筛选：涨跌幅、量比、换手率、PE、市值等 |
| **📈 策略回测** | 6 种内置策略，事件驱动引擎，输出完整绩效报告 |
| **💡 每日建议** | 综合评分 + 操作建议 + 止损止盈位 + 风险提示 |
| **📦 组合跟踪** | 输入持仓列表，自动计算盈亏并给出操作建议 |
| **🌐 MCP 接口** | 标准 MCP 协议，Claude / Cursor / Cline / Codex 等 Agent 直接调用 |

## 🏗️ 架构设计

```
astock-agent/
├── src/
│   ├── __init__.py          # 包初始化
│   ├── data_provider.py     # 数据层 - AKShare 统一封装
│   ├── analyzer.py          # 分析层 - 六维度智能分析引擎
│   ├── strategies.py        # 策略层 - 6种内置交易策略
│   ├── backtest.py          # 回测层 - 事件驱动回测引擎
│   ├── mcp_server.py        # Agent接口层 - MCP Server (9个工具)
│   └── cli.py               # CLI入口 - 命令行直接使用
├── data/                    # 数据缓存目录
├── tests/                   # 测试用例
├── pyproject.toml           # 项目配置
└── README.md                # 本文档
```

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/your-username/astock-agent.git
cd astock-agent

# 创建虚拟环境(推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 依赖说明

| 依赖包 | 用途 |
|--------|------|
| `akshare` | A股数据源（行情/资金流/财报/舆情） |
| `pandas` / `numpy` | 数据分析与计算 |
| `mcp[cli]` | MCP Server 协议支持 |
| `ta-lib` | 技术指标计算库 |
| `rich` | 终端美化输出 |
| `click` | CLI 框架 |
| `pydantic` | 数据校验 |

## 🎯 使用方式

### 方式一：MCP Server（推荐给 Agent 使用）

在 Claude Desktop / Cursor / Cline 等 AI 客户端中配置：

```json
{
  "mcpServers": {
    "astock-agent": {
      "command": "python",
      "args": ["-m", "mcp", "run", "src/mcp_server.py"],
      "cwd": "/path/to/astock-agent"
    }
  }
}
```

配置完成后，Agent 即可调用以下 **9 个工具**：

| 工具名 | 功能 | 示例调用 |
|--------|------|----------|
| `select_stocks` | 条件选股 | "帮我选出今天放量上涨的低PE股票" |
| `analyze_stock` | 单股六维深度分析 | "分析一下贵州茅台" |
| `batch_analyze` | 批量分析多只股票 | "对比分析茅台、五粮液、老窖" |
| `get_quote` | 实时行情查询 | "看看宁德时代现在的价格" |
| `get_money_flow` | 资金流向分析 | "比亚迪的主力资金流向如何" |
| `backtest_strategy` | 策略回测 | "对宁德时代做MACD策略回测" |
| `daily_advice` | 每日操作建议 ⭐ | "给我今天的操作建议" |
| `track_portfolio` | 持仓组合跟踪 | "跟踪我的持仓组合" |
| `get_sector_data` | 行业板块排行 | "今天哪些行业表现好" |

#### Agent 对话示例

> **用户**: 帮我分析一下 600519（贵州茅台），给出今天的操作建议
>
> **Agent**: [调用 `daily_advice(symbol="600519")`] 
>
> 分析结果：
> - 📊 **综合评分: 68.5 分**（看好）
> - 技术面: 72分 — 均线多头排列，MACD金叉
> - 估值: 55分 — PE-TTM=28.5，处于近3年45%分位
> - 资金面: 75分 — 主力净流入3.2亿，资金做多意愿强
> - 基本面: 70分 — ROE=28%，营收同比+18%
> - 财报质量: 65分 — 经营现金流覆盖净利润
> - 舆情: 60分 — 近7日新闻情绪偏正面
>
> **建议**: 综合评估良好，可持有或轻仓参与。当前价1680元，参考止损1546元(-8%)，参考止盈1932元(+15%)

---

### 方式二：CLI 命令行（适合终端使用）

```bash
# 分析单只股票
python -m src.cli analyze 600519

# 批量分析对比
python -m src.cli analyze 600519 000858 300750

# 条件选股（放量上涨+低PE）
python -m src.cli screen --min-change 1 --max-change 5 --max-pe 30 --min-vol-ratio 1.2

# 策略回测
python -m src.cli backtest 600519 macd           # 单策略
python -m src.cli backtest 600519 all             # 全部策略对比

# 每日操作建议（最常用）
python -m src.cli advice 600519
python -m src.cli advice 600519 --with-backtest   # 附带回测验证

# 实时行情
python -m src.cli quote 600519
python -m src.cli quote 600519,000858,300750      # 多只

# 资金流向
python -m src.cli moneyflow 600519

# 行业板块排行
python -m src.cli sector

# JSON格式输出（方便程序处理）
python -m src.cli analyze 600519 --json
```

## 🔬 六维度分析体系详解

### 1️⃣ 技术面 (权重 20%)
- **均线系统**: MA5/MA10/MA20/MA60 多头/空头排列
- **MACD**: DIF/DEA 金叉死叉 + 柱状线方向 + 底背离检测
- **RSI**: 14日 RSI 超买超卖判断
- **KDJ**: K/D 金叉死叉 + 位置判断
- **布林带**: 价格在上/下轨位置 + 收口/开口状态
- **成交量**: 量比、放量上涨/下跌确认

### 2️⃣ 估值 (权重 18%)
- **PE-TTM**: 绝对值区间评估
- **PB**: 净资产溢价水平
- **历史分位**: 近3年/5年 PE/PB 百分位
- **PEG**: 估值与成长性匹配度
- **股息率**: 估算分红收益率

### 3️⃣ 资金面 (权重 22%)
- **主力净流入/流出**: 金额及占成交额比例
- **超大单/大单/中单/小单**: 各层级资金动向
- **龙虎榜**: 近两月上榜记录及机构净买卖
- **散户反向指标**: 小单流入 vs 主力流出

### 4️⃣ 基本面 (权重 18%)
- **盈利能力**: ROE、净利率、毛利率
- **成长性**: 营收/净利润同比增速
- **偿债能力**: 资产负债率、流动比率、速动比率
- **现金流**: 每股经营现金流 vs EPS

### 5️⃣ 财报质量 (权重 10%)
- **利润现金流匹配度**: OCF vs Net Income
- **ROE驱动来源**: 经营效率 vs 杠杆
- **收入质量**: 净利率异常检测
- **运营效率**: 总资产周转率

### 6️⃣ 舆情情绪 (权重 12%)
- **新闻情绪方向**: 正面/负面/中性/混合
- **关键词强度**: 利好/利空关键词统计
- **市场关注度**: 新闻数量反映热度
- **换手率补充**: 交易活跃度与投机氛围

## 📈 内置策略一览

| 策略 | 名称 | 适用场景 | 核心逻辑 |
|------|------|----------|----------|
| `dual_ma` | 双均线交叉 | 趋势跟踪 | MA5上穿/下穿MA20 |
| `macd` | MACD经典 | 震荡+趋势 | DIF/DEA金叉死叉+柱状线确认 |
| `bollinger` | 布林带突破 | 波动率突破 | 价格突破上下轨+回归信号 |
| `grid` | 网格交易 | 震荡市场 | 固定网格间距低买高卖 |
| `momentum` | 动量突破 | 趋势启动 | 放量突破N日高低点 |
| `turtle` | 海龟交易法 | 趋势跟踪 | 20日通道突破+10日离市+ATR止损 |

### 回测输出指标

每次回测返回完整的绩效报告：
- **总收益率 / 年化收益率**
- **最大回撤**
- **夏普比率**（年化，无风险利率3%）
- **胜率 & 盈亏比**
- **详细交易记录**（每笔买入/卖出/盈亏）

## ⚙️ 配置说明

### 数据缓存
默认缓存有效期 **5 分钟**，可在代码中调整：

```python
provider = AStockDataProvider(cache_ttl=300)  # 秒
provider.clear_cache()  # 手动清空缓存
```

### 回测参数
```python
engine = BacktestEngine(
    initial_capital=100000.0,   # 初始资金10万
    commission_rate=0.0003,      # 万三佣金
    slippage=0.001,             # 千一滑点
    stamp_tax=0.001,            # 卖出千一印花税
)
```

## 🧪 开发与扩展

### 添加自定义策略

```python
from strategies import BaseStrategy, Signal, STRATEGY_REGISTRY

class MyStrategy(BaseStrategy):
    """自定义策略"""
    
    def generate_signals(self, kline_df):
        df = kline_df.copy()
        close = df["close"].astype(float)
        # ... 你的策略逻辑 ...
        df["action"] = "BUY"  # 或 "SELL" / "HOLD"
        df["strength"] = 0.8
        df["reason"] = "你的策略描述"
        return df
    
    def get_current_signal(self, kline_df):
        df = self.generate_signals(kline_df)
        last = df.iloc[-1]
        return Signal(
            action=last["action"],
            strength=last["strength"],
            reason=last["reason"],
        )

# 注册到全局注册表
STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

### 添加新的分析维度

```python
from analyzer import DimensionScore, StockAnalyzer

class MyDimensionAnalyzer:
    @staticmethod
    def analyze(data) -> DimensionScore:
        score = 50.0  # 你的评分逻辑
        return DimensionScore(
            name="我的维度",
            score=score,
            details={...},
            signals=[...],
        )

# 在 StockAnalyzer.analyze() 中集成
```

## ⚠️ 免责声明

> 本工具仅为**技术学习和研究目的**提供，所有数据和分析结果仅供参考，**不构成任何投资建议**。
> 
> 股市有风险，投资需谨慎。过往回测表现不代表未来收益。使用者应基于自身独立判断做出投资决策。
> 
> 本项目开发者不对因使用本工具导致的任何投资损失承担责任。

## 📄 开源协议

MIT License — 自由使用、修改和分发

## 🙏 致谢

- [AKShare](https://github.com/akfamily/akshare) — 优秀的 A股开源数据接口
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io) — Agent 工具标准协议
- [vn.py](https://github.com/vnpy/vnpy) — 国产量化框架参考
- [Qlib](https://github.com/microsoft/qlib) — 微软AI量化平台参考

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

*Made with ❤️ for A股投资者*

</div>
