"""
事件驱动回测引擎
支持: 多策略回测 / 绩效指标计算 / 权益曲线 / 交易记录导出
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime

try:
    from .strategies import STRATEGY_REGISTRY, get_strategy
except ImportError:  # 直接以脚本方式运行 src/ 目录时
    from strategies import STRATEGY_REGISTRY, get_strategy


@dataclass
class TradeRecord:
    """单笔交易记录"""
    symbol: str
    action: str  # "BUY" / "SELL"
    price: float
    quantity: int
    timestamp: str
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    initial_capital: float = 100000.0
    final_capital: float = 100000.0
    equity_curve: List[float] = field(default_factory=list)
    trade_records: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    error: Optional[str] = None


class BacktestEngine:
    """事件驱动回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,   # 万三佣金
        slippage: float = 0.001,            # 滑点 0.1%
        stamp_tax: float = 0.001,           # 印花税(卖出千一)
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.stamp_tax = stamp_tax

    def run(
        self,
        kline_df: pd.DataFrame,
        strategy,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        运行回测

        Args:
            kline_df: K线数据(需含 date/open/high/low/close/volume)
            strategy: 策略实例(BaseStrategy子类)
            symbol: 股票代码

        Returns:
            BacktestResult
        """
        if kline_df is None or kline_df.empty or len(kline_df) < 30:
            return BacktestResult(
                symbol=symbol,
                strategy_name=strategy.name if hasattr(strategy, 'name') else "unknown",
                start_date="",
                end_date="",
                error="数据不足",
            )

        # 生成信号
        signal_df = strategy.generate_signals(kline_df.copy())
        if "action" not in signal_df.columns:
            return BacktestResult(
                symbol=symbol,
                strategy_name=getattr(strategy, 'name', 'unknown'),
                start_date="",
                end_date="",
                error="策略未生成信号",
            )

        # 回测状态
        capital = self.initial_capital
        position = 0          # 持仓数量
        entry_price = 0.0     # 开仓价
        equity_curve = []
        trade_records = []
        daily_returns = []

        prev_equity = capital

        for idx, row in signal_df.iterrows():
            close_price = float(row["close"])
            date_str = str(row.get("date", ""))
            action = row.get("action", "HOLD")
            reason = row.get("reason", "")

            # 计算当前权益
            if position > 0:
                equity = capital + position * close_price
            else:
                equity = capital

            # 执行交易信号
            if action == "BUY" and position == 0:
                # 买入(考虑滑点)
                buy_price = close_price * (1 + self.slippage)
                max_shares = int(capital / (buy_price * (1 + self.commission_rate)))
                if max_shares >= 100:  # 至少一手
                    cost = max_shares * buy_price
                    commission = cost * self.commission_rate
                    capital -= (cost + commission)
                    position = max_shares
                    entry_price = buy_price

                    trade_records.append({
                        "date": date_str,
                        "action": "BUY",
                        "price": round(buy_price, 2),
                        "quantity": max_shares,
                        "commission": round(commission, 2),
                        "reason": reason,
                    })

            elif action == "SELL" and position > 0:
                # 卖出(考虑滑点+印花税)
                sell_price = close_price * (1 - self.slippage)
                revenue = position * sell_price
                commission = revenue * self.commission_rate
                tax = revenue * self.stamp_tax
                capital += (revenue - commission - tax)

                # 记录盈亏
                cost_basis = position * entry_price
                pnl = revenue - cost_basis - commission - tax
                pnl_pct = (sell_price / entry_price - 1) * 100

                trade_records[-1]["exit_date"] = date_str
                trade_records[-1]["exit_price"] = round(sell_price, 2)
                trade_records[-1]["pnl"] = round(pnl, 2)
                trade_records[-1]["pnl_pct"] = round(pnl_pct, 2)

                position = 0
                entry_price = 0.0

            # 更新权益曲线
            if position > 0:
                current_equity = capital + position * close_price
            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # 日收益率
            if prev_equity > 0:
                daily_ret = (current_equity / prev_equity - 1) * 100
                daily_returns.append(daily_ret)
            prev_equity = current_equity

        # 最终平仓(如果还有持仓)
        if position > 0:
            last_close = float(signal_df.iloc[-1]["close"])
            sell_price = last_close * (1 - self.slippage)
            revenue = position * sell_price
            commission = revenue * self.commission_rate
            tax = revenue * self.stamp_tax
            capital += (revenue - commission - tax)

            cost_basis = position * entry_price
            pnl = revenue - cost_basis - commission - tax
            pnl_pct = (sell_price / entry_price - 1) * 100

            trade_records.append({
                "date": str(signal_df.iloc[-1].get("date", "")),
                "action": "SELL(FORCED)",
                "price": round(sell_price, 2),
                "quantity": position,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "reason": "回测结束强制平仓",
            })
            position = 0

        # 计算绩效指标
        result = self._calculate_metrics(
            symbol=symbol,
            strategy_name=getattr(strategy, 'name', 'unknown'),
            start_date=str(signal_df.iloc[0].get("date", "")) if len(signal_df) > 0 else "",
            end_date=str(signal_df.iloc[-1].get("date", "")) if len(signal_df) > 0 else "",
            initial_capital=self.initial_capital,
            final_capital=capital,
            equity_curve=equity_curve,
            trade_records=trade_records,
            daily_returns=daily_returns,
        )

        return result

    def _calculate_metrics(
        self,
        symbol: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        final_capital: float,
        equity_curve: List[float],
        trade_records: List[Dict],
        daily_returns: List[float],
    ) -> BacktestResult:
        """计算回测绩效指标"""

        total_return = (final_capital / initial_capital - 1) * 100 if initial_capital > 0 else 0

        # 年化收益
        if start_date and end_date:
            try:
                d_start = pd.to_datetime(start_date)
                d_end = pd.to_datetime(end_date)
                days = (d_end - d_start).days
                if days > 0:
                    years = days / 365.25
                    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100
                else:
                    annualized_return = total_return
            except (ValueError, TypeError, ZeroDivisionError):
                annualized_return = total_return
        else:
            annualized_return = total_return

        # 最大回撤
        peak = initial_capital
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # 夏普比率(无风险利率按3%年化)
        if daily_returns and len(daily_returns) > 5:
            returns_arr = np.array(daily_returns)
            avg_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr)
            if std_ret > 0:
                # 日均收益年化
                sharpe = (avg_ret * 252 - 3) / (std_ret * np.sqrt(252))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # 胜率与盈亏比
        completed_trades = [t for t in trade_records if "pnl" in t]
        winning = [t for t in completed_trades if t.get("pnl", 0) > 0]
        losing = [t for t in completed_trades if t.get("pnl", 0) <= 0]

        win_rate = len(winning) / len(completed_trades) * 100 if completed_trades else 0

        avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losing])) if losing else None
        profit_loss_ratio = avg_win / avg_loss if avg_loss and avg_loss > 0 else 0

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            win_rate=round(win_rate, 1),
            profit_loss_ratio=round(profit_loss_ratio, 2),
            total_trades=len(completed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            equity_curve=[round(e, 2) for e in equity_curve],
            trade_records=trade_records,
            daily_returns=[round(r, 3) for r in daily_returns],
        )


def run_multi_strategy_backtest(
    kline_df: pd.DataFrame,
    symbol: str,
    strategies: list = None,
) -> List[BacktestResult]:
    """多策略对比回测

    Args:
        kline_df: K线数据
        symbol: 股票代码
        strategies: 策略列表，默认运行全部内置策略

    Returns:
        各策略回测结果列表
    """
    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    engine = BacktestEngine()
    results = []

    for strat_name in strategies:
        try:
            strat_instance = get_strategy(strat_name)
            result = engine.run(kline_df, strat_instance, symbol=symbol)
            results.append(result)
        except Exception as e:
            results.append(BacktestResult(
                symbol=symbol,
                strategy_name=strat_name,
                start_date="", end_date="",
                error=str(e),
            ))

    return results


def format_backtest_report(result: BacktestResult) -> str:
    """格式化回测报告为可读文本"""
    lines = [
        f"\n{'='*60}",
        f"  回测报告: {result.symbol} | {result.strategy_name}",
        f"{'='*60}",
        f"  回测区间: {result.start_date} ~ {result.end_date}",
        f"  初始资金: ¥{result.initial_capital:,.2f}",
        f"  最终资金: ¥{result.final_capital:,.2f}",
        f"{'─'*60}",
        f"  总收益率:      {result.total_return_pct:+.2f}%",
        f"  年化收益率:    {result.annualized_return_pct:+.2f}%",
        f"  最大回撤:      {result.max_drawdown_pct:.2f}%",
        f"  夏普比率:      {result.sharpe_ratio:.3f}",
        f"  胜率:          {result.win_rate:.1f}%",
        f"  盈亏比:        {result.profit_loss_ratio:.2f}",
        f"  总交易次数:    {result.total_trades}",
        f"  盈利次数:      {result.winning_trades}",
        f"  亏损次数:      {result.losing_trades}",
        f"{'='*60}",
    ]

    if result.trade_records:
        lines.append("\n  交易记录:")
        lines.append(f"  {'日期':<12} {'操作':<14} {'价格':>10} {'数量':>8} {'盈亏':>12}")
        lines.append(f"  {'─'*60}")
        for t in result.trade_records[:20]:  # 最多显示20条
            date = t.get("date", "")
            action = t.get("action", "")
            price = t.get("price", 0)
            qty = t.get("quantity", 0)
            pnl_str = f"{t.get('pnl', 0):+.2f}" if "pnl" in t else "-"
            lines.append(f"  {date:<12} {action:<14} {price:>10.2f} {qty:>8} {pnl_str:>12}")

        if len(result.trade_records) > 20:
            lines.append(f"  ... 还有 {len(result.trade_records)-20} 条记录")

    return "\n".join(lines)
