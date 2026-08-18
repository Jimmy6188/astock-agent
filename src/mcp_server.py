"""
MCP Server - A股智能分析工具的Agent接口层
暴露给LLM Agent(Claude/Cursor/Cline/Codex等)调用的标准工具集

启动方式:
  python -m mcp run src/mcp_server.py    # 标准MCP模式
  uvx mcp run src/mcp_server.py          # 通过uvx运行

工具列表:
  select_stocks     - 条件选股(多维度筛选)
  analyze_stock     - 单股六维深度分析
  batch_analyze     - 批量分析多只股票
  get_quote         - 获取实时行情
  get_money_flow    - 获取资金流向
  backtest_strategy - 策略回测
  daily_advice      - 每日操作建议(综合)
  track_portfolio   - 持仓跟踪与建议
  get_sector_data   - 行业板块数据
"""

import asyncio
import json
import logging
from typing import Any, Optional
from mcp.server import Server
from mcp.types import Tool, TextContent

# 导入核心模块
from .data_provider import AStockDataProvider, get_provider, set_meta_recorder
from .analyzer import StockAnalyzer
from .strategies import STRATEGY_REGISTRY, get_strategy
from .backtest import BacktestEngine, run_multi_strategy_backtest, format_backtest_report

logger = logging.getLogger(__name__)

# 创建MCP Server实例
server = Server("astock-agent")

# ==================== 元数据上下文 ====================

# 全局元数据：记录每次调用的数据源状态
_META = {}

def _meta_snapshot():
    """拷贝当前元数据快照"""
    return dict(_META)

def _record(source: str = "", fallback: bool = False, note: str = ""):
    """记录本次调用的数据源元信息"""
    if source:
        _META["sources"] = _META.get("sources", []) + [source]
    if fallback:
        _META["fallbacks"] = _META.get("fallbacks", []) + [note or source]

# 注入元数据记录器（使data_provider的_record_source生效）
set_meta_recorder(_record)
provider = get_provider()
analyzer = StockAnalyzer(provider)


# ==================== 工具注册 ====================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用工具"""
    return [
        Tool(
            name="select_stocks",
            description="""A股条件选股工具。根据多个维度筛选符合条件的股票。
支持筛选条件: 涨跌幅范围、量比、换手率、PE上限、最小市值等。
返回: 符合条件的股票列表(代码/名称/价格/涨跌幅/成交额等)。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_change_pct": {
                        "type": "number",
                        "description": "最低涨跌幅%(默认-10)",
                        "default": -10.0,
                    },
                    "max_change_pct": {
                        "type": "number",
                        "description": "最高涨跌幅%(默认10)",
                        "default": 10.0,
                    },
                    "min_volume_ratio": {
                        "type": "number",
                        "description": "最低量比(默认0.8)",
                        "default": 0.8,
                    },
                    "min_turnover": {
                        "type": "number",
                        "description": "最低换手率%(默认0.5)",
                        "default": 0.5,
                    },
                    "max_pe": {
                        "type": "number",
                        "description": "PE-TTM上限(默认100，0表示不限制)",
                        "default": 100.0,
                    },
                    "min_market_cap": {
                        "type": "number",
                        "description": "最小市值(亿元，默认10)",
                        "default": 10.0,
                    },
                },
            },
        ),
        Tool(
            name="analyze_stock",
            description="""对单只A股进行完整的六维度智能分析。
六个维度: 技术面(均线/MACD/RSI/KDJ/布林带/成交量)、估值(PE/PB分位/PEG)、
资金面(主力净流入/龙虎榜)、基本面(ROE/成长性/偿债能力)、财报质量(现金流匹配度)、舆情情绪。
输出: 综合评分(0~100)、各维度评分、关键信号、操作建议、风险提示。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码(6位数字，如000001/600519)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="batch_analyze",
            description="""批量分析多只A股股票。每只股票执行完整六维分析。
适合用于: 自选股批量评估、行业对比、组合扫描等场景。
返回: 每只股票的综合评分和操作建议列表。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "股票代码列表，如['000001','600519','300750']",
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_quote",
            description="""获取单只或多只A股实时行情数据。
包含: 最新价、涨跌幅、成交量、成交额、换手率、PE/PB、总市值、流通市值等。
可传入单个代码或代码列表。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码(6位)，如600519。或逗号分隔的多只代码",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_money_flow",
            description="""获取个股资金流向数据。
包含: 主力净流入/流出金额及占比、超大单/大单/中单/小单资金流、近两月龙虎榜记录。
用于判断主力资金动向和市场情绪。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码(6位)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="backtest_strategy",
            description="""对指定股票执行策略回测。
支持的策略: dual_ma(双均线交叉), macd(MACD经典), bollinger(布林带突破),
grid(网格交易), momentum(动量突破), turtle(海龟交易法)。
输出: 总收益率、年化收益、最大回撤、夏普比率、胜率、盈亏比、详细交易记录。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码(6位)",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "策略名称: dual_ma/macd/bollinger/grid/momentum/turtle/all",
                        "default": "all",
                    },
                    "period": {
                        "type": "string",
                        "description": "K线周期: daily/weekly/monthly",
                        "default": "daily",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "回测起始日期(YYYYMMDD，默认1年前)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "回测结束日期(YYYYMMDD，默认今天)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="daily_advice",
            description="""生成每日操作建议报告。综合技术面、资金面、估值、基本面、舆情等多维度数据，
给出具体的买入/卖出/持有建议，附带止损位和止盈位参考。
这是最常用的日常工具，输入股票代码即可获得完整的投资决策参考。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码(6位)，如600519",
                    },
                    "include_backtest": {
                        "type": "boolean",
                        "description": "是否同时执行快速回测验证(默认false)",
                        "default": False,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="track_portfolio",
            description="""持仓组合跟踪与分析。输入持仓列表(股票代码+数量+成本价)，
计算当前盈亏、给出每只持仓的操作建议、整体风险评估。
适合每日盘后复盘使用。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "quantity": {"type": "number"},
                                "cost_price": {"type": "number"},
                            },
                            "required": ["symbol", "quantity", "cost_price"],
                        },
                        "description": "持仓列表，每项含代码/数量/成本价",
                    },
                },
                "required": ["holdings"],
            },
        ),
        Tool(
            name="get_sector_data",
            description="""获取A股行业板块涨跌排行数据。
返回所有行业的涨跌幅、成交额、领涨/领跌个股等信息。
用于判断市场热点板块和轮动方向。""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ==================== 工具实现 ====================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """处理工具调用"""
    try:
        if name == "select_stocks":
            result = _select_stocks(**arguments)

        elif name == "analyze_stock":
            result = _analyze_stock(**arguments)

        elif name == "batch_analyze":
            result = _batch_analyze(**arguments)

        elif name == "get_quote":
            result = _get_quote(**arguments)

        elif name == "get_money_flow":
            result = _get_money_flow(**arguments)

        elif name == "backtest_strategy":
            result = await _backtest_strategy(**arguments)

        elif name == "daily_advice":
            result = await _daily_advice(**arguments)

        elif name == "track_portfolio":
            result = await _track_portfolio(**arguments)

        elif name == "get_sector_data":
            result = _get_sector_data()

        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"未知工具: {name}",
                "available_tools": [t.name for t in (await list_tools())],
            }, ensure_ascii=False, indent=2))]

        # 统一JSON序列化输出
        meta = _meta_snapshot()
        _META.clear()  # 清空本次的元数据
        if isinstance(result, dict):
            result = _attach_meta(result, meta)
        elif isinstance(result, list):
            result = _attach_meta({"results": result}, meta)

        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return [TextContent(type="text", text=output)]

    except Exception as e:
        logger.error(f"工具 {name} 执行异常: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "tool": name,
        }, ensure_ascii=False, indent=2))]


# ==================== 工具实现函数 ====================

def _select_stocks(
    min_change_pct: float = -10.0,
    max_change_pct: float = 10.0,
    min_volume_ratio: float = 0.8,
    min_turnover: float = 0.5,
    max_pe: float = 100.0,
    min_market_cap: float = 10.0,
) -> dict:
    """条件选股"""
    results = provider.screen_stocks(
        min_change_pct=min_change_pct,
        max_change_pct=max_change_pct,
        min_volume_ratio=min_volume_ratio,
        min_turnover=min_turnover,
        max_pe=max_pe,
        min_market_cap=min_market_cap,
    )
    return {
        "action": "screen_stocks",
        "criteria": {
            "min_change_pct": min_change_pct,
            "max_change_pct": max_change_pct,
            "min_volume_ratio": min_volume_ratio,
            "min_turnover": min_turnover,
            "max_pe": max_pe,
            "min_market_cap_yi": min_market_cap,
        },
        "count": len(results),
        "results": results[:50],  # 最多返回50只
        "timestamp": pd_timestamp_now(),
    }


def _analyze_stock(symbol: str) -> dict:
    """单股六维分析"""
    report = analyzer.analyze(symbol)
    return report


def _batch_analyze(symbols: list) -> dict:
    """批量分析"""
    results = analyzer.batch_analyze(symbols)
    # 按综合得分排序
    results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return {
        "action": "batch_analyze",
        "total": len(symbols),
        "analyzed": len([r for r in results if "error" not in r]),
        "ranked_results": results,
        "timestamp": pd_timestamp_now(),
    }


def _get_quote(symbol: str) -> dict:
    """获取行情"""
    # 支持逗号分隔的多只代码
    symbols = [s.strip() for s in symbol.split(",")]
    if len(symbols) == 1:
        result = provider.get_realtime_quote(symbols[0])
        return {"action": "quote", "result": result}
    else:
        results = provider.get_batch_quotes(symbols)
        return {"action": "batch_quote", "count": len(results), "results": results}


def _get_money_flow(symbol: str) -> dict:
    """资金流向"""
    mf = provider.get_money_flow(symbol)
    lhb = provider.get_lhb(symbol)
    return {
        "action": "money_flow",
        "symbol": symbol,
        "money_flow": mf,
        "lhb_records": lhb,
        "timestamp": pd_timestamp_now(),
    }


async def _backtest_strategy(
    symbol: str,
    strategy: str = "all",
    period: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """策略回测"""
    kline = provider.get_history_kline(
        symbol, period=period,
        start_date=start_date, end_date=end_date,
    )

    if kline.empty:
        return {"error": f"无法获取{symbol}的K线数据", "symbol": symbol}

    if strategy == "all":
        strategies_to_run = list(STRATEGY_REGISTRY.keys())
    else:
        strategies_to_run = [s.strip() for s in strategy.split(",")]

    results = run_multi_strategy_backtest(kline, symbol, strategies_to_run)

    report_data = []
    for r in results:
        if hasattr(r, 'error') and r.error:
            report_data.append({"strategy": r.strategy_name, "error": r.error})
        else:
            report_data.append({
                "strategy": r.strategy_name,
                "total_return_pct": r.total_return_pct,
                "annualized_return_pct": r.annualized_return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "win_rate": r.win_rate,
                "profit_loss_ratio": r.profit_loss_ratio,
                "total_trades": r.total_trades,
                "final_capital": r.final_capital,
                "trade_count_summary": f"{r.winning_trades}胜/{r.losing_trades}负",
            })

    # 推荐最佳策略
    valid_results = [r for r in results if not (hasattr(r, 'error') and r.error)]
    best = max(valid_results, key=lambda x: x.total_return_pct) if valid_results else None

    return {
        "action": "backtest",
        "symbol": symbol,
        "period": period,
        "date_range": f"{start_date or '1年前'} ~ {end_date or '今天'}",
        "strategies_tested": len(strategies_to_run),
        "results": report_data,
        "best_strategy": best.strategy_name if best else None,
        "best_return_pct": best.total_return_pct if best else None,
        "note": "回测结果仅供参考，不构成投资建议。实盘需考虑滑点、冲击成本、流动性等因素。",
        "timestamp": pd_timestamp_now(),
    }


async def _daily_advice(symbol: str, include_backtest: bool = False) -> dict:
    """每日操作建议（最常用工具）"""
    # 六维分析
    report = analyzer.analyze(symbol)

    advice = {
        "action": "daily_advice",
        "symbol": symbol,
        "date": pd_timestamp_now()[:10],
        "summary": {
            "overall_score": report.get("overall_score"),
            "rating": report.get("overall_rating"),
            "action_suggestion": report.get("action_suggestion"),
        },
        "dimensions": report.get("dimensions", []),
        "key_signals": report.get("key_signals", []),
        "risk_warnings": report.get("risk_warnings", []),
    }

    # 附带最新行情快照
    quote = provider.get_realtime_quote(symbol)
    if isinstance(quote, dict) and "error" not in quote:
        advice["quote_snapshot"] = {
            "name": quote.get("name", ""),
            "price": quote.get("price"),
            "change_pct": quote.get("change_pct"),
            "volume_ratio": quote.get("volume_ratio"),
            "turnover_rate": quote.get("turnover_rate"),
            "amount_yi": round(quote.get("amount", 0) / 1e8, 2),
        }

    # 通达信指标综合分析
    tdx_data = report.get("tdx_indicators")
    if tdx_data:
        advice["tdx_indicators"] = {
            "verdict": tdx_data.get("combined_verdict"),
            "confidence": tdx_data.get("confidence"),
            "recommendation": tdx_data.get("recommendation"),
            "signals": tdx_data.get("signals", []),
        }

    # 可选：快速回测验证
    if include_backtest:
        kline = provider.get_history_kline(symbol, period="daily")
        if not kline.empty:
            bt_result = run_multi_strategy_backtest(kline, symbol, ["dual_ma", "macd"])
            advice["quick_backtest"] = [{
                "strategy": r.strategy_name,
                "return_pct": r.total_return_pct,
                "sharpe": r.sharpe_ratio,
                "win_rate": r.win_rate,
            } for r in bt_result if not (hasattr(r, 'error') and r.error)]

    return advice


async def _track_portfolio(holdings: list) -> dict:
    """持仓跟踪与操作建议 - 增强版(含持仓视角分析/多策略共振/多空辩论)"""
    holdings_analysis = []
    total_cost = 0
    total_value = 0
    total_pnl = 0

    from position_analyzer import PositionAnalyzer
    pa = PositionAnalyzer()

    for h in holdings:
        sym = h["symbol"]
        qty = h["quantity"]
        cost = h["cost_price"]

        quote = provider.get_realtime_quote(sym)
        analysis = analyzer.analyze(sym)

        if isinstance(quote, dict) and "error" not in quote:
            price = quote.get("price", 0)
            value = price * qty
            pnl = (price - cost) * qty
            pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0

            total_cost += cost * qty
            total_value += value
            total_pnl += pnl

            # 持仓视角分析
            pos_advice = pa.analyze({
                "symbol": sym, "quantity": qty, "cost_price": cost,
                "current_price": price, "pnl_pct": pnl_pct,
            }, stock_score={
                "overall_score": analysis.get("overall_score"),
                "overall_rating": analysis.get("overall_rating"),
                "key_signals": analysis.get("key_signals", []),
            })

            # 多策略共振
            resonance_data = analysis.get("resonance") or {}

            # 多空辩论
            debate_data = analysis.get("debate") or {}

            # 缠论
            chan_data = analysis.get("chan") or {}

            # 价值投资
            vi_data = analysis.get("value_investing") or {}

            holdings_analysis.append({
                "symbol": sym,
                "name": quote.get("name", ""),
                "quantity": qty,
                "cost_price": cost,
                "current_price": price,
                "market_value": round(value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                # 持仓视角
                "position_status": pos_advice.position_status,
                "health_score": pos_advice.health_score,
                "health_label": pos_advice.health_label,
                "action": pos_advice.action,
                "action_detail": pos_advice.action_detail,
                "target_price": pos_advice.target_price,
                "stop_loss": pos_advice.stop_loss,
                # 六维分析
                "score": analysis.get("overall_score"),
                "rating": analysis.get("overall_rating"),
                "suggestion": analysis.get("action_suggestion", "").split("\n")[0] if analysis.get("action_suggestion") else "",
                # 扩展分析摘要
                "resonance": resonance_data,
                "debate_verdict": debate_data.get("verdict", ""),
                "chan_position": chan_data.get("position", ""),
                "chan_buy": chan_data.get("buy_point"),
                "vi_verdict": vi_data.get("value_verdict", ""),
            })
        else:
            holdings_analysis.append({
                "symbol": sym, "error": "无法获取行情数据"
            })

    portfolio_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # 分类统计
    healthy = [h for h in holdings_analysis if h.get("health_label") == "健康"]
    risky = [h for h in holdings_analysis if h.get("health_label") == "危险"]
    urgent_sell = [h for h in holdings_analysis if h.get("action") in ("减仓", "止盈", "换仓")]

    return {
        "action": "portfolio_track",
        "holdings_count": len(holdings),
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(portfolio_pnl_pct, 2),
        "healthy_count": len(healthy),
        "risky_count": len(risky),
        "urgent_action_count": len(urgent_sell),
        "holdings": holdings_analysis,
        "risk_assessment": _assess_portfolio_risk(holdings_analysis),
        "timestamp": pd_timestamp_now(),
    }


def _get_sector_data() -> dict:
    """行业板块数据"""
    df = provider.get_sector_performance()
    if df.empty:
        return {"error": "无法获取板块数据"}

    # 取前20名和后10名
    df_sorted = df.sort_values("涨跌幅", ascending=False)
    top = df_sorted.head(20)
    bottom = df_sorted.tail(10)

    def row_to_dict(r):
        return {
            "sector": str(r.get("板块名称", "")),
            "change_pct": float(r.get("涨跌幅", 0)) if pd.notna(r.get("涨跌幅")) else 0,
            "amount_yi": round(float(r.get("成交额", 0)) / 1e8, 2) if pd.notna(r.get("成交额")) else 0,
            "up_count": int(r.get("上涨家数", 0)) if pd.notna(r.get("上涨家数")) else 0,
            "down_count": int(r.get("下跌家数", 0)) if pd.notna(r.get("下跌家数")) else 0,
        }

    return {
        "action": "sector_data",
        "top_gainers": [row_to_dict(r) for _, r in top.iterrows()],
        "top_losers": [row_to_dict(r) for _, r in bottom.iterrows()],
        "timestamp": pd_timestamp_now(),
    }


# ==================== 辅助函数 ====================

def _attach_meta(payload, meta):
    """将 _meta 字段附加到结果payload中"""
    if not meta:
        return payload
    payload = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(payload, dict):
        payload["_meta"] = meta
    return payload


def pd_timestamp_now() -> str:
    try:
        from datetime import datetime
        return datetime.now().isoformat()
    except Exception:
        return ""


def _assess_portfolio_risk(holdings: list) -> str:
    """简单组合风险评估"""
    warnings = []
    losers = [h for h in holdings if h.get("pnl_pct", 0) < -10]
    high_concentration = [h for h in holdings if h.get("market_value", 0) > sum(h.get("market_value", 0) for h in holdings) * 0.4]

    if losers:
        warnings.append(f"{len(losers)}只持仓亏损超过10%")
    if high_concentration:
        warnings.append(f"持仓集中度过高: {high_concentration[0]['symbol']}占比超40%")

    low_score = [h for h in holdings if h.get("score", 50) < 35]
    if low_score:
        warnings.append(f"{len(low_score)}只持仓综合评分偏低(<35)")

    if not warnings:
        return "组合风险可控，继续保持关注"
    return "风险提示: " + "; ".join(warnings)


# ==================== 启动入口 ====================

async def _run_stdio():
    """使用 mcp v1.29.0 stdio transport 启动"""
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


def main():
    """MCP Server 启动入口(通过 mcp CLI 或直接运行)"""
    import asyncio
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
