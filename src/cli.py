"""
A股智能分析工具 CLI 入口
支持命令行直接运行，无需 MCP Server

用法:
  python -m src.cli analyze 600519          # 分析单只股票
  python -m src.cli analyze 600519 300750   # 批量分析
  python -m src.cli screen                   # 条件选股
  python -m src.cli backtest 600519 macd     # 策略回测
  python -m src.cli advice 600519            # 每日操作建议
  python -m src.cli portfolio                # 持仓跟踪(需配置)
  python -m src.cli sector                   # 行业板块数据
"""

import sys
import json
import pandas as pd
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

# 导入核心模块
from .data_provider import get_provider
from .analyzer import StockAnalyzer
from .strategies import STRATEGY_REGISTRY, get_strategy
from .backtest import BacktestEngine, run_multi_strategy_backtest, format_backtest_report


def _json_output(data: dict):
    """JSON格式输出"""
    console.print_json(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _pretty_analyze(report: dict):
    """美化输出分析报告"""
    # 头部信息
    console.print(Panel(
        f"[bold cyan]{report.get('symbol', '?')}[/] | "
        f"综合评分: [bold {score_color(report.get('overall_score', 0))}]{report.get('overall_score', 0):.1f}分[/] | "
        f"[bold]{report.get('overall_rating', '')}[/]\n"
        f"{report.get('action_suggestion', '')}",
        title="📊 A股六维智能分析报告",
        border_style="cyan",
    ))

    # 维度评分表
    table = Table(title="维度评分详情", show_header=True, header_style="bold magenta")
    table.add_column("维度", style="cyan")
    table.add_column("评分", justify="center")
    table.add_column("评级", justify="center")
    table.add_column("关键信号")

    for dim in report.get("dimensions", []):
        signals_str = "\n".join(dim.get("signals", [])[:3])
        if len(dim.get("signals", [])) > 3:
            signals_str += f"\n...还有{len(dim['signals'])-3}条信号"
        table.add_row(
            dim["dimension"],
            f"[bold {score_color(dim['score'])}]{dim['score']:.1f}[/]",
            dim["rating"],
            signals_str or "-",
        )
    console.print(table)

    # 关键信号
    if report.get("key_signals"):
        console.print("\n[bold yellow]🔑 关键信号:[/]")
        for s in report["key_signals"][:10]:
            console.print(f"  • {s}")

    # 风险提示
    if report.get("risk_warnings"):
        console.print("\n[bold red]⚠️ 风险提示:[/]")
        for w in report["risk_warnings"]:
            console.print(f"  • {w}")


def score_color(score: float) -> str:
    """根据分数返回颜色"""
    if score >= 75: return "green"
    if score >= 55: return "yellow"
    if score >= 35: return "bright_black"
    return "red"


# ==================== CLI 命令 ====================

@click.group()
@click.version_option(version="1.0.0", prog_name="astock-agent")
def main():
    """🚀 A股智能分析Agent工具 - 实时选股、多维分析、回测与策略制定"""
    pass


@main.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--json", "json_output", is_flag=True, help="以JSON格式输出")
def analyze(symbols, json_output):
    """分析单只或多只股票（六维深度分析）"""
    provider = get_provider()
    analyzer = StockAnalyzer(provider)

    sym_list = list(symbols)

    if len(sym_list) == 1:
        report = analyzer.analyze(sym_list[0])
        if json_output:
            _json_output(report)
        else:
            _pretty_analyze(report)
    else:
        results = analyzer.batch_analyze(sym_list)
        if json_output:
            _json_output({"results": results})
        else:
            # 排行榜表格
            table = Table(title=f"批量分析结果 ({len(sym_list)}只)", show_header=True)
            table.add_column("排名", justify="center", width=4)
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            table.add_column("综合分", justify="center")
            table.add_column("评级", justify="center")
            table.add_column("建议")

            for i, r in enumerate(sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)):
                if "error" in r:
                    continue
                suggestion = r.get("action_suggestion", "").split("\n")[0] if r.get("action_suggestion") else ""
                table.add_row(
                    str(i + 1),
                    r.get("symbol", ""),
                    r.get("data_snapshot", {}).get("quote", {}).get("name", ""),
                    f"[{score_color(r.get('overall_score', 0))}]{r.get('overall_score', 0):.1f}[/]",
                    r.get("overall_rating", ""),
                    suggestion[:30],
                )
            console.print(table)


@main.command()
@click.option("--min-change", type=float, default=-10.0, help="最低涨跌幅%")
@click.option("--max-change", type=float, default=10.0, help="最高涨跌幅%")
@click.option("--min-vol-ratio", type=float, default=0.8, help="最低量比")
@click.option("--min-turnover", type=float, default=0.5, help="最低换手率%")
@click.option("--max-pe", type=float, default=100.0, help="PE上限")
@click.option("--min-cap", type=float, default=10.0, help="最小市值(亿)")
@click.option("--json", "json_output", is_flag=True, help="JSON格式输出")
def screen(min_change, max_change, min_vol_ratio, min_turnover, max_pe, min_cap, json_output):
    """条件选股"""
    provider = get_provider()
    results = provider.screen_stocks(
        min_change_pct=min_change,
        max_change_pct=max_change,
        min_volume_ratio=min_vol_ratio,
        min_turnover=min_turnover,
        max_pe=max_pe,
        min_market_cap=min_cap,
    )

    if json_output:
        _json_output({"count": len(results), "results": results})
    else:
        table = Table(title=f"选股结果 ({len(results)}只)", show_header=True)
        table.add_column("#", justify="center", width=4)
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("现价", justify="right")
        table.add_column("涨跌幅%", justify="right")
        table.add_column("量比", justify="right")
        table.add_column("换手%", justify="right")
        table.add_column("成交额(亿)", justify="right")

        for i, r in enumerate(results[:30]):
            chg_color = "green" if r.get("change_pct", 0) >= 0 else "red"
            table.add_row(
                str(i + 1),
                r["symbol"],
                r["name"],
                f"{r['price']:.2f}",
                f"[{chg_color}]{r['change_pct']:+.2f}[/]",
                f"{r['volume_ratio']:.2f}",
                f"{r['turnover_rate']:.2f}",
                f"{r['amount']/1e8:.1f}" if r.get("amount") else "-",
            )
        console.print(table)
        if len(results) > 30:
            console.print(f"\n... 还有 {len(results)-30} 只符合条件")


@main.command()
@click.argument("symbol")
@click.option("--strategy", "strategies", default="all",
              help="策略: all/dual_ma/macd/bollinger/grid/momentum/turtle")
@click.option("--period", default="daily", help="周期: daily/weekly/monthly")
@click.option("--json", "json_output", is_flag=True, help="JSON格式输出")
def backtest(symbol, strategies, period, json_output):
    """策略回测"""
    provider = get_provider()
    kline = provider.get_history_kline(symbol, period=period)

    if kline.empty:
        console.print(f"[red]无法获取 {symbol} 的K线数据[/]")
        return

    strat_list = list(STRATEGY_REGISTRY.keys()) if strategies == "all" else [s.strip() for s in strategies.split(",")]
    results = run_multi_strategy_backtest(kline, symbol, strat_list)

    if json_output:
        output = []
        for r in results:
            d = {
                "strategy": r.strategy_name,
                "total_return_pct": r.total_return_pct,
                "annualized_return": r.annualized_return_pct,
                "max_drawdown": r.max_drawdown_pct,
                "sharpe": r.sharpe_ratio,
                "win_rate": r.win_rate,
                "trades": r.total_trades,
            }
            output.append(d)
        _json_output(output)
    else:
        for r in results:
            if hasattr(r, 'error') and r.error:
                console.print(f"[yellow]{r.strategy_name}: {r.error}[/]")
                continue
            console.print(format_backtest_report(r))


@main.command()
@click.argument("symbol")
@click.option("--with-backtest", is_flag=True, help="同时执行快速回测")
@click.option("--json", "json_output", is_flag=True, help="JSON格式输出")
def advice(symbol, with_backtest, json_output):
    """每日操作建议（最常用）"""
    provider = get_provider()
    analyzer = StockAnalyzer(provider)

    report = analyzer.analyze(symbol)

    advice_data = {
        "symbol": symbol,
        "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
        "overall_score": report.get("overall_score"),
        "rating": report.get("overall_rating"),
        "suggestion": report.get("action_suggestion"),
        "dimensions": report.get("dimensions"),
        "signals": report.get("key_signals")[:10],
        "warnings": report.get("risk_warnings"),
    }

    # 行情快照
    quote = provider.get_realtime_quote(symbol)
    if isinstance(quote, dict) and "error" not in quote:
        advice_data["quote"] = {
            "name": quote.get("name"),
            "price": quote.get("price"),
            "change_pct": quote.get("change_pct"),
        }

    if with_backtest:
        kline = provider.get_history_kline(symbol, period="daily")
        if not kline.empty:
            bt = run_multi_strategy_backtest(kline, symbol, ["dual_ma", "macd"])
            advice_data["backtest"] = [{
                "strat": r.strategy_name,
                "return": r.total_return_pct,
                "sharpe": r.sharpe_ratio,
            } for r in bt if not (hasattr(r, 'error') and r.error)]

    if json_output:
        _json_output(advice_data)
    else:
        _pretty_analyze(report)


@main.command()
@click.option("--json", "json_output", is_flag=True, help="JSON格式输出")
def sector(json_output):
    """行业板块排行"""
    provider = get_provider()
    df = provider.get_sector_performance()

    if df.empty:
        console.print("[red]无法获取板块数据[/]")
        return

    if json_output:
        _json_output(df.to_dict(orient="records"))
    else:
        df_sorted = df.sort_values("涨跌幅", ascending=False)
        table = Table(title="行业板块涨跌排行", show_header=True)
        table.add_column("#", justify="center", width=4)
        table.add_column("板块名称", style="cyan")
        table.add_column("涨跌幅%", justify="right")
        table.add_column("成交额(亿)", justify="right")
        table.add_column("上涨/下跌家数", justify="center")

        for i, (_, r) in enumerate(df_sorted.head(30).iterrows()):
            chg = r.get("涨跌幅", 0)
            chg_color = "green" if (pd.notna(chg) and chg >= 0) else "red"
            up = int(r.get("上涨家数", 0)) if pd.notna(r.get("上涨家数")) else 0
            down = int(r.get("下跌家数", 0)) if pd.notna(r.get("下跌家数")) else 0
            amount = r.get("成交额", 0)
            table.add_row(
                str(i + 1),
                str(r.get("板块名称", "")),
                f"[{chg_color}]{chg:+.2f}[/]" if pd.notna(chg) else "-",
                f"{amount/1e8:.1f}" if pd.notna(amount) else "-",
                f"{up}/{down}",
            )
        console.print(table)


@main.command()
@click.argument("symbol")
def quote(symbol):
    """查看实时行情"""
    provider = get_provider()
    symbols = [s.strip() for s in symbol.split(",")]

    if len(symbols) == 1:
        q = provider.get_realtime_quote(symbols[0])
        if isinstance(q, dict) and "error" not in q:
            table = Table(title=f"{q.get('name', '')} ({q.get('symbol', '')})", show_header=False)
            table.add_column("指标", style="cyan", width=14)
            table.add_column("值", justify="right")
            rows = [
                ("最新价", f"{q.get('price', 0):.2f}"),
                ("涨跌幅", f"{q.get('change_pct', 0):+.2f}%"),
                ("涨跌额", f"{q.get('change_amt', 0):+.2f}"),
                ("今开", f"{q.get('open', 0):.2f}"),
                ("最高", f"{q.get('high', 0):.2f}"),
                ("最低", f"{q.get('low', 0):.2f}"),
                ("成交量", f"{q.get('volume', 0):,.0f}"),
                ("成交额", f"{q.get('amount', 0)/1e8:.2f}亿"),
                ("量比", (lambda vr: f"{vr:.2f}" if isinstance(vr, (int, float)) else "-")(q.get('volume_ratio'))),
                ("换手率", f"{q.get('turnover_rate', 0):.2f}%"),
                ("PE-TTM", f"{q.get('pe_ttm', '-')}"),
                ("PB", f"{q.get('pb', '-')}"),
                ("总市值", f"{q.get('total_mv', 0)/1e8:.1f}亿"),
                ("流通市值", f"{q.get('circ_mv', 0)/1e8:.1f}亿"),
            ]
            for label, val in rows:
                table.add_row(label, val)
            console.print(table)
        else:
            console.print(f"[red]{q}[/]")
    else:
        results = provider.get_batch_quotes(symbols)
        table = Table(show_header=True)
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("价格")
        table.add_column("涨跌幅%")
        table.add_column("换手%")
        for r in results:
            if "error" not in r:
                chg_color = "green" if r.get("change_pct", 0) >= 0 else "red"
                table.add_row(
                    r["symbol"], r["name"],
                    f"{r['price']:.2f}",
                    f"[{chg_color}]{r['change_pct']:+.2f}[/]",
                    f"{r.get('turnover_rate', 0):.2f}",
                )
        console.print(table)


@main.command()
@click.argument("symbol")
def moneyflow(symbol):
    """查看资金流向"""
    provider = get_provider()
    mf = provider.get_money_flow(symbol)
    lhb = provider.get_lhb(symbol)

    if isinstance(mf, dict) and "error" not in mf:
        table = Table(title=f"{symbol} 资金流向", show_header=False)
        table.add_column("指标", style="cyan", width=18)
        table.add_column("值", justify="right")
        table.add_row("日期", str(mf.get("date", "")))
        table.add_row("主力净流入(万)", f"{mf.get('main_net_inflow', 0)/10000:,.1f}")
        table.add_row("主力净流入占比", f"{mf.get('main_net_pct', 0):+.2f}%")
        table.add_row("超大单净额(万)", f"{mf.get('super_large_net', 0)/10000:,.1f}")
        table.add_row("大单净额(万)", f"{mf.get('large_net', 0)/10000:,.1f}")
        table.add_row("中单净额(万)", f"{mf.get('medium_net', 0)/10000:,.1f}")
        table.add_row("小单净额(万)", f"{mf.get('small_net', 0)/10000:,.1f}")
        console.print(table)

    if lhb:
        console.print(f"\n[bold]近两月龙虎榜记录({len(lhb)}次):[/]")
        for record in lhb[:5]:
            console.print(f"  {record.get('date', '')} | {record.get('reason', '')} | "
                         f"净买入:{record.get('net_buy', 0)/10000:.0f}万")
    elif isinstance(mf, dict) and "error" in mf:
        console.print(f"[red]{mf['error']}[/]")


if __name__ == "__main__":
    main()
