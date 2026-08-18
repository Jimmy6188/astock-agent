"""
多策略共振引擎 (Multi-Strategy Resonance Engine)
灵感来源: ai-hedge-fund / ai-hedge-fund-framework-2026

核心思想: 当多个独立策略给出同向信号时，信号可信度显著提升。
6个策略(dual_ma, macd, bollinger, momentum, turtle, grid)各自产生信号，
共振引擎检测信号一致性、方向统一性、时间窗口集中度。

输出:
  - resonance_score: 0~100，越高代表策略间共识越强
  - dominant_direction: BUY / SELL / HOLD
  - convergent_signals: 哪些策略一致
  - divergent_signals: 哪些策略分歧
  - confidence: 综合置信度
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .strategies import (
    STRATEGY_REGISTRY, get_strategy, Signal,
    DualMAStrategy, MACDStrategy, BollingerBreakoutStrategy,
    MomentumBreakoutStrategy, TurtleStrategy, GridTradingStrategy,
)
from .backtest import run_multi_strategy_backtest


@dataclass
class ResonanceResult:
    resonance_score: float          # 0~100
    dominant_direction: str         # BUY / SELL / HOLD
    confidence: str                 # 高/中/低
    convergent_signals: List[str]   # 一致的策略名
    divergent_signals: List[Dict]   # 分歧的策略详情
    signal_counts: Dict[str, int]   # {BUY: n, SELL: m, HOLD: k}
    backtest_rank: Dict[str, float] # 策略回测表现排序
    recommendation: str             # 综合建议
    timestamp: str = ""


class ResonanceEngine:
    """多策略共振分析引擎"""

    STRATEGY_ORDER = ["dual_ma", "macd", "bollinger", "momentum", "turtle", "grid"]

    def analyze(self, kline_df: pd.DataFrame, symbol: str,
                lookback: int = 30) -> ResonanceResult:
        """对给定K线数据执行多策略共振分析

        Args:
            kline_df: 历史K线(pandas DataFrame, 含date/open/high/low/close/volume列)
            symbol: 股票代码
            lookback: 回看窗口(用于检测信号时间集中度)

        Returns:
            ResonanceResult
        """
        from datetime import datetime

        if kline_df is None or kline_df.empty:
            return ResonanceResult(
                resonance_score=0, dominant_direction="HOLD",
                confidence="低", convergent_signals=[], divergent_signals=[],
                signal_counts={"BUY": 0, "SELL": 0, "HOLD": 0},
                backtest_rank={}, recommendation="无K线数据",
                timestamp=datetime.now().isoformat(),
            )

        # 1. 获取每个策略当前信号
        signals = {}
        for name in self.STRATEGY_ORDER:
            try:
                strat = STRATEGY_REGISTRY.get(name) or get_strategy(name)
                sig = strat.get_current_signal(kline_df)
                signals[name] = sig.action if hasattr(sig, 'action') else 'HOLD'
            except Exception:
                signals[name] = 'HOLD'

        # 2. 统计信号分布
        buy_count = sum(1 for v in signals.values() if v == 'BUY')
        sell_count = sum(1 for v in signals.values() if v == 'SELL')
        hold_count = sum(1 for v in signals.values() if v == 'HOLD')
        total = len(signals)

        # 3. 计算共振分
        max_same = max(buy_count, sell_count, hold_count)
        resonance = (max_same / total) * 100

        # 4. 判断主导方向
        if max_same == buy_count and buy_count >= 3:
            direction = "BUY"
        elif max_same == sell_count and sell_count >= 3:
            direction = "SELL"
        elif buy_count == sell_count:
            direction = "HOLD"
        elif buy_count > sell_count:
            direction = "BUY"
        elif sell_count > buy_count:
            direction = "SELL"
        else:
            direction = "HOLD"

        # 5. 置信度
        if resonance >= 80:
            confidence = "高"
        elif resonance >= 60:
            confidence = "中"
        else:
            confidence = "低"

        # 6. 一致策略
        convergent = [name for name, act in signals.items() if act == direction]

        # 7. 分歧策略
        divergent = []
        for name, act in signals.items():
            if act != direction:
                divergent.append({"strategy": name, "signal": act})

        # 8. 快速回测排名
        bt_rank = self._quick_backtest_rank(kline_df, symbol)

        # 9. 综合建议
        rec = self._build_recommendation(
            direction, resonance, confidence, convergent, divergent, bt_rank
        )

        return ResonanceResult(
            resonance_score=round(resonance, 1),
            dominant_direction=direction,
            confidence=confidence,
            convergent_signals=convergent,
            divergent_signals=divergent,
            signal_counts={"BUY": buy_count, "SELL": sell_count, "HOLD": hold_count},
            backtest_rank=bt_rank,
            recommendation=rec,
            timestamp=datetime.now().isoformat(),
        )

    def _quick_backtest_rank(self, kline_df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """快速回测6策略，按收益排序"""
        try:
            results = run_multi_strategy_backtest(kline_df, symbol, self.STRATEGY_ORDER)
            ranking = {}
            for r in results:
                if not (hasattr(r, 'error') and r.error):
                    ranking[r.strategy_name] = round(r.total_return_pct, 2)
            return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}

    def _build_recommendation(self, direction: str, resonance: float,
                               confidence: str, convergent: List[str],
                               divergent: List[Dict], bt_rank: Dict) -> str:
        if resonance >= 80 and confidence == "高":
            return f"多策略强共振({resonance:.0f}%),{direction}信号,建议{self._dir_action(direction)}"
        elif resonance >= 60:
            return f"多策略中等共振({resonance:.0f}%),{direction}倾向,可{self._dir_action(direction)}"
        elif bt_rank:
            best = list(bt_rank.keys())[0]
            best_ret = list(bt_rank.values())[0]
            return f"策略分歧,但{best}回测收益{best_ret}%,可参考该策略"
        else:
            return "策略信号分散,建议观望或轻仓试探"

    def _dir_action(self, direction: str) -> str:
        return {"BUY": "建仓/加仓", "SELL": "减仓/止损", "HOLD": "持有观望"}.get(direction, "观望")