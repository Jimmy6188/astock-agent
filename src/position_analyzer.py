"""
持仓视角分析引擎 (Position-Aware Analysis Engine)

核心思想: 传统的六维分析是"市场视角"——只看股票本身好坏。
但实际操作中，"好不好"取决于你持有它的状态。

持仓视角分析:
  1. 盈亏状态感知 (盈利/亏损/深套)
  2. 动态止盈止损建议 (基于盈亏比例)
  3. 仓位建议 (加仓/减仓/换仓)
  4. 持仓健康度评分

适用场景: "持仓股复盘"时，对每只持仓股执行。
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from .analyzer import StockAnalyzer


@dataclass
class PositionAdvice:
    stock_name: str
    symbol: str
    quantity: int
    cost_price: float
    current_price: float
    pnl: float
    pnl_pct: float

    # 持仓状态
    position_status: str        # "深套" / "亏损" / "微亏" / "保本" / "微盈" / "盈利" / "大盈"
    health_score: float         # 0~100 持仓健康度
    health_label: str           # "健康" / "亚健康" / "危险"

    # 操作建议
    action: str                 # "加仓" / "持有" / "减仓" / "止盈" / "换仓" / "观望"
    action_detail: str          # 详细建议
    target_price: float         # 目标价(止盈/止损位)
    stop_loss: float            # 止损价

    # 分析数据
    score: Optional[float]      # 六维综合得分
    rating: str                 # 评级
    signals: List[str]          # 关键信号


class PositionAnalyzer:
    """持仓视角分析器"""

    def analyze(self, holding: dict, stock_score: Optional[dict] = None) -> PositionAdvice:
        """对单只持仓执行分析

        Args:
            holding: 持仓数据(symbol/quantity/cost_price/current_price等)
            stock_score: 可选的六维分析结果(如果已计算)

        Returns:
            PositionAdvice
        """
        sym = holding['symbol']
        name = holding.get('name', sym)
        qty = holding['quantity']
        cost = holding.get('cost_price', 0)
        cur = holding.get('current_price', 0)
        pnl = holding.get('pnl', (cur - cost) * qty)
        pnl_pct = holding.get('pnl_pct', ((cur - cost) / cost * 100) if cost else 0)

        score = stock_score.get('overall_score') if stock_score else None
        rating = stock_score.get('overall_rating', '') if stock_score else ''
        signals = stock_score.get('key_signals', []) if stock_score else []

        # 1. 持仓状态
        status = self._position_status(pnl_pct)

        # 2. 持仓健康度
        health = self._health_score(pnl_pct, score, qty, cost)
        health_label = "健康" if health >= 70 else ("亚健康" if health >= 40 else "危险")

        # 3. 操作建议
        action, detail, target, stop_loss = self._generate_advice(
            status, health, score, rating, signals, pnl_pct, cur, cost,
        )

        return PositionAdvice(
            stock_name=name,
            symbol=sym,
            quantity=qty,
            cost_price=cost,
            current_price=cur,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            position_status=status,
            health_score=round(health, 1),
            health_label=health_label,
            action=action,
            action_detail=detail,
            target_price=round(target, 2),
            stop_loss=round(stop_loss, 2),
            score=score,
            rating=rating,
            signals=signals[:5],
        )

    def _position_status(self, pnl_pct: float) -> str:
        if pnl_pct > 50:
            return "大盈"
        elif pnl_pct > 10:
            return "盈利"
        elif pnl_pct > 2:
            return "微盈"
        elif pnl_pct > -2:
            return "保本"
        elif pnl_pct > -10:
            return "微亏"
        elif pnl_pct > -30:
            return "亏损"
        else:
            return "深套"

    def _health_score(self, pnl_pct: float, score: Optional[float],
                       qty: int, cost: float) -> float:
        health = 50

        # 盈亏贡献 (0~40)
        if pnl_pct > 0:
            health += min(pnl_pct / 5, 40)
        else:
            health += max(pnl_pct / 5, -40)

        # 六维得分贡献 (0~30)
        if score is not None:
            health += (score - 50) * 0.3

        # 负成本特殊加分 (做T成功)
        if cost < 0:
            health += 20

        return max(0, min(100, health))

    def _generate_advice(self, status: str, health: float,
                          score: Optional[float], rating: str,
                          signals: list, pnl_pct: float,
                          cur: float, cost: float) -> tuple:
        """生成操作建议"""

        # 大盈: 考虑止盈
        if status == "大盈":
            if score and score >= 60:
                return (
                    "持有观察",
                    f"大盈{pnl_pct:.0f}%且信号积极({score}分),继续持有但设好止盈",
                    round(cur * 1.15, 2),   # 止盈位+15%
                    round(cur * 0.85, 2),   # 止损位-15%
                )
            elif score and score < 40:
                return (
                    "减仓止盈",
                    f"大盈{pnl_pct:.0f}%但信号转弱({score}分),建议减仓一半锁定利润",
                    round(cur * 0.92, 2),   # 回撤到92%止盈
                    round(cur * 0.80, 2),
                )
            else:
                return (
                    "部分止盈",
                    f"大盈{pnl_pct:.0f}%,建议止盈50%锁定利润,剩余继续持有",
                    round(cur * 1.20, 2),
                    round(cur * 0.85, 2),
                )

        # 盈利: 持有
        elif status == "盈利":
            if score and score >= 55:
                return (
                    "持有",
                    f"盈利{pnl_pct:.0f}%且信号良好({score}分),继续持有",
                    round(cur * 1.25, 2),
                    round(cur * 0.90, 2),
                )
            else:
                return (
                    "持有观望",
                    f"盈利{pnl_pct:.0f}%但信号中性({score}分),持有观望",
                    round(cur * 1.20, 2),
                    round(cur * 0.85, 2),
                )

        # 微盈/保本: 观望
        elif status in ("微盈", "保本"):
            if score and score >= 55:
                return (
                    "持有",
                    f"保本状态且信号良好({score}分),继续持有",
                    round(cur * 1.20, 2),
                    round(cur * 0.85, 2),
                )
            elif score and score < 40:
                return (
                    "观望/减仓",
                    f"微利状态但信号偏弱({score}分),设好止损",
                    round(cur * 1.15, 2),
                    round(cost * 0.90, 2),
                )
            else:
                return (
                    "持有观望",
                    "当前微利/保本,信号中性,持有观望",
                    round(cur * 1.20, 2),
                    round(cost * 0.90, 2),
                )

        # 微亏: 可考虑加仓
        elif status == "微亏":
            if score and score >= 55:
                return (
                    "加仓",
                    f"微亏{pnl_pct:.0f}%但信号良好({score}分),可适量加仓摊低成本",
                    round(float('nan'), 2) if True else 0,
                    round(cost * 0.85, 2),
                )
            else:
                return (
                    "观望",
                    f"微亏{pnl_pct:.0f}%且信号中性,暂不加仓",
                    round(cost * 1.15, 2),
                    round(cost * 0.85, 2),
                )

        # 亏损: 判断是否基本面恶化
        elif pnl_pct > -30:
            if 'score' in dir() and score is not None:
                if score >= 55:
                    return (
                        "补仓摊低",
                        f"亏损{pnl_pct:.0f}%但信号良好({score}分),可分批补仓摊低",
                        round(cost * 1.10, 2),
                        round(cost * 0.80, 2),
                    )
                elif score < 40:
                    return (
                        "考虑止损",
                        f"亏损{pnl_pct:.0f}%且信号偏弱({score}分),评估是否止损",
                        round(cost * 1.10, 2),
                        round(cost * 0.85, 2),
                    )
            return (
                "持有观望",
                f"亏损{pnl_pct:.0f}%,持有观望,设好止损",
                round(cost * 1.10, 2),
                round(cost * 0.80, 2),
            )

        # 深套: 最后建议
        else:
            if 'score' in dir() and score and score >= 55:
                return (
                    "补仓+耐心",
                    f"深套{pnl_pct:.0f}%但基本面尚可({score}分),分批补仓+耐心等反弹",
                    round(cost * 1.05, 2),
                    round(cost * 0.70, 2),
                )
            else:
                return (
                    "换仓或长期持有",
                    f"深套{pnl_pct:.0f}%,若基本面恶化考虑换仓,否则长期持有等周期",
                    round(cost * 1.00, 2),
                    round(cost * 0.70, 2),
                )