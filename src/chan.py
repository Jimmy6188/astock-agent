"""
缠论笔段分析引擎 (Chan Theory Analysis Engine)
灵感来源: easy_tdx (668 stars, 通达信协议直连)

核心概念:
  - 分型: 顶分型(K线组中间高) / 底分型(K线组中间低)
  - 笔: 相邻两个有效分型之间的线段(至少3根K线)
  - 中枢: 三段重叠区间
  - 买卖点: 1买(底分型确认), 2买(回抽不破前低), 3买(突破中枢)

简化实现:
  1. 检测顶/底分型
  2. 生成"笔"结构
  3. 识别当前所处位置(顶/底/上升/下降)
  4. 输出简化买卖点判断
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ChanType:
    """缠论分型"""
    kind: str         # "top" 或 "bottom"
    price: float      # 分型价格
    index: int        # 在K线中的位置
    strength: str     # "强" / "中" / "弱"


@dataclass
class ChanPen:
    """缠论笔"""
    start: ChanType
    end: ChanType
    length: int
    direction: str    # "up" 或 "down"


@dataclass
class ChanCenter:
    """缠论中枢"""
    top: float
    bottom: float
    pen_count: int


@dataclass
class ChanAnalysisResult:
    types: List[ChanType]
    pens: List[ChanPen]
    centers: List[ChanCenter]
    current_position: str   # "上升段" / "下降段" / "震荡中枢" / "趋势不明"
    buy_point: Optional[str]  # "一买" / "二买" / "三买" / None
    sell_point: Optional[str] # "一卖" / "二卖" / "三卖" / None
    buy_price: Optional[float]   # 具体买入参考价
    sell_price: Optional[float]  # 具体卖出参考价
    structure_score: float  # 0~100, 结构越清晰分数越高
    recommendation: str
    warnings: List[str]


class ChanAnalyzer:
    """缠论笔段分析器(简化版)"""

    MIN_PEN_LENGTH = 5       # 最小笔长度(K线数)
    MIN_OVERLAP = 0          # 中枢最小重叠

    def analyze(self, kline_df: pd.DataFrame, lookback: int = 60) -> ChanAnalysisResult:
        """执行缠论分析

        Args:
            kline_df: 历史K线(需含date/open/high/low/close/volume列)
            lookback: 分析窗口(天)

        Returns:
            ChanAnalysisResult
        """
        if kline_df is None or kline_df.empty or len(kline_df) < lookback:
            return ChanAnalysisResult(
                types=[], pens=[], centers=[],
                current_position="趋势不明",
                buy_point=None, sell_point=None,
                buy_price=None, sell_price=None,
                structure_score=0, recommendation="数据不足，无法判断",
                warnings=["K线数据不足"],
            )

        # 取最近N天
        df = kline_df.tail(lookback).copy().reset_index(drop=True)
        highs = df['high'].values if 'high' in df.columns else None
        lows = df['low'].values if 'low' in df.columns else None
        closes = df['close'].values if 'close' in df.columns else None

        if highs is None or lows is None:
            return ChanAnalysisResult(
                types=[], pens=[], centers=[],
                current_position="趋势不明", buy_point=None, sell_point=None,
                buy_price=None, sell_price=None,
                structure_score=0, recommendation="缺少高低点数据",
                warnings=["K线数据格式不符"],
            )

        # 1. 检测分型
        types = self._detect_types(highs, lows)

        # 2. 生成笔
        pens = self._make_pens(types, lookback)

        # 3. 检测中枢
        centers = self._detect_centers(pens)

        # 4. 判断当前位置
        pos = self._current_position(pens, closes)

        # 5. 买卖点
        buy, buy_price = self._buy_point(pens, centers, closes)
        sell, sell_price = self._sell_point(pens, centers, closes)

        # 6. 结构分数
        score = self._structure_score(pens, centers, types, lookback)

        # 7. 建议
        rec = self._recommendation(pos, buy, sell, score, buy_price, sell_price)

        warnings = []
        if len(types) < 3:
            warnings.append("分型较少，缠论结构尚未形成")
        if len(pens) < 2:
            warnings.append("笔数不足，难以判断趋势结构")

        return ChanAnalysisResult(
            types=types, pens=pens, centers=centers,
            current_position=pos, buy_point=buy, sell_point=sell,
            buy_price=buy_price, sell_price=sell_price,
            structure_score=round(score, 1), recommendation=rec,
            warnings=warnings,
        )

    def _detect_types(self, highs: np.ndarray, lows: np.ndarray) -> List[ChanType]:
        """检测顶底分型"""
        types = []
        n = len(highs)

        for i in range(1, n - 1):
            # 顶分型: 中间K线高点最高
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                strength = self._type_strength(highs, i, 'top')
                # 去重: 跳过连续顶分型
                if not types or types[-1].kind != 'top':
                    types.append(ChanType('top', float(highs[i]), i, strength))
                else:
                    # 取更高的
                    if highs[i] > types[-1].price:
                        types[-1] = ChanType('top', float(highs[i]), i, strength)

            # 底分型: 中间K线低点最低
            elif lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                strength = self._type_strength(lows, i, 'bottom')
                if not types or types[-1].kind != 'bottom':
                    types.append(ChanType('bottom', float(lows[i]), i, strength))
                else:
                    if lows[i] < types[-1].price:
                        types[-1] = ChanType('bottom', float(lows[i]), i, strength)

        return types

    def _type_strength(self, values: np.ndarray, i: int, kind: str) -> str:
        """判断分型强弱"""
        if kind == 'top':
            diff = values[i] - max(values[i-1], values[i+1])
            avg_range = (values[i+1] - values[i-1]) if values[i+1] > values[i-1] else (values[i-1] - values[i+1])
        else:
            diff = min(values[i-1], values[i+1]) - values[i]
            avg_range = (values[i+1] - values[i-1]) if values[i+1] > values[i-1] else (values[i-1] - values[i+1])

        if avg_range <= 0:
            return "弱"
        ratio = diff / avg_range
        if ratio > 0.5:
            return "强"
        elif ratio > 0.2:
            return "中"
        return "弱"

    def _make_pens(self, types: List[ChanType], lookback: int) -> List[ChanPen]:
        """从分型生成笔"""
        pens = []
        # 需要至少顶底交替
        for i in range(len(types) - 1):
            if types[i].kind == types[i+1].kind:
                continue
            length = types[i+1].index - types[i].index
            if length < self.MIN_PEN_LENGTH:
                continue
            if types[i].kind == 'bottom':
                direction = 'up'
            else:
                direction = 'down'
            pens.append(ChanPen(
                start=types[i], end=types[i+1],
                length=length, direction=direction,
            ))

        return pens

    def _detect_centers(self, pens: List[ChanPen]) -> List[ChanCenter]:
        """检测中枢(简化: 连续3笔中有重叠区间)"""
        centers = []
        for i in range(len(pens) - 2):
            p1, p2, p3 = pens[i], pens[i+1], pens[i+2]
            # 找到重叠区间
            segments = [
                (p2.start.price, p2.end.price),
                (p3.start.price, p3.end.price),
            ]
            seg_lo = max(min(s) for s in segments)
            seg_hi = min(max(s) for s in segments)
            if seg_hi > seg_lo:
                centers.append(ChanCenter(top=seg_hi, bottom=seg_lo, pen_count=3))

        return centers

    def _current_position(self, pens: List[ChanPen], closes: np.ndarray) -> str:
        """判断当前处于什么位置"""
        if not pens:
            return "趋势不明"

        last_pen = pens[-1]
        if last_pen.direction == 'up':
            return "上升段"
        elif last_pen.direction == 'down':
            return "下降段"
        return "震荡"

    def _buy_point(self, pens: List[ChanPen], centers: List[ChanCenter],
                   closes: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """识别买点,返回(标签, 价格)"""
        if not pens or len(pens) < 2:
            return None, None

        last_pen = pens[-1]
        prev_pen = pens[-2]

        # 一买: 下降趋势中的底分型确认
        if last_pen.direction == 'up' and prev_pen.direction == 'down':
            last_bottom = last_pen.start.price
            if closes[-1] > last_bottom:
                return "一买(底部反转)", round(last_bottom, 2)

        # 二买: 一买后回抽不破前低
        if len(pens) >= 3:
            pen3 = pens[-3]
            if pen3.direction == 'down' and last_pen.direction == 'up':
                prev_low = pen3.end.price
                if prev_low and last_pen.start.price > prev_low:
                    return "二买(回抽确认)", round(last_pen.start.price, 2)

        # 三买: 突破中枢后回踩不破中枢上沿
        if centers and last_pen.direction == 'up':
            center = centers[-1]
            if closes[-1] > center.top * 1.02:
                return "三买(突破回踩)", round(center.bottom, 2)

        return None, None

    def _sell_point(self, pens: List[ChanPen], centers: List[ChanCenter],
                    closes: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """识别卖点,返回(标签, 价格)"""
        if not pens or len(pens) < 2:
            return None, None

        last_pen = pens[-1]
        prev_pen = pens[-2]

        if last_pen.direction == 'down' and prev_pen.direction == 'up':
            return "一卖(顶部反转)", round(last_pen.start.price, 2)

        if len(pens) >= 3:
            pen3 = pens[-3]
            if pen3.direction == 'up' and last_pen.direction == 'down':
                prev_high = pen3.end.price
                if prev_high and last_pen.start.price < prev_high:
                    return "二卖(回抽确认)", round(last_pen.start.price, 2)

        if centers and last_pen.direction == 'down':
            center = centers[-1]
            if closes[-1] < center.bottom * 0.98:
                return "三卖(破位确认)", round(center.top, 2)

        return None, None

    def _structure_score(self, pens: List[ChanPen], centers: List[ChanCenter],
                         types: List[ChanType], lookback: int) -> float:
        """计算缠论结构清晰度分数"""
        score = 0
        max_score = 100

        # 分型数量 (0~30)
        type_ratio = min(len(types) / 10, 1.0)
        score += type_ratio * 30

        # 笔数 (0~30)
        pen_ratio = min(len(pens) / 6, 1.0)
        score += pen_ratio * 30

        # 中枢数量 (0~20)
        center_ratio = min(len(centers) / 2, 1.0)
        score += center_ratio * 20

        # 趋势连续性 (0~20)
        if len(pens) >= 3:
            directions = [p.direction for p in pens[-3:]]
            if len(set(directions)) == 1:
                score += 15  # 同方向连续，趋势清晰
            else:
                score += 5

        return min(score, max_score)

    def _recommendation(self, position: str, buy: Optional[str], sell: Optional[str],
                        score: float, buy_price: Optional[float], sell_price: Optional[float]) -> str:
        """生成缠论建议"""
        if score < 30:
            return "缠论结构尚未成形，建议结合其他分析方法"

        if buy:
            price_str = f"，参考价{buy_price}元" if buy_price else ""
            return f"缠论提示:{buy}{price_str} | 当前处于{position} | 结构清晰度{score}分"
        elif sell:
            price_str = f"，参考价{sell_price}元" if sell_price else ""
            return f"缠论提示:{sell}{price_str} | 当前处于{position} | 结构清晰度{score}分"
        else:
            return f"缠论无明确买卖点 | 当前处于{position} | 结构清晰度{score}分"