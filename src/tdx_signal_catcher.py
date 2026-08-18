"""
通达信「提前抄底」信号翻译模块

将通达信副图指标「提前抄底」翻译为 Python。包含 12 个独立信号：

  1. 多方力度 vs 空方力度 — 强弱对比（核心）
  2. 一阳定势 — 一阳包多均线确认趋势反转
  3. 红钻石 — 放量阳线 + 多方金叉
  4. 金银袋 — 回测均线不破 + 多方金叉
  5. 龙抬头 — KDJ 金叉 + 趋势安全区
  6. 钻石底 — 价格大幅偏离 40 日均线
  7. 抄底 — WR 双周期极端超卖
  8. 逃顶 — WR 双周期极端超买
  9. 主力吸筹 — 底部区域吸筹量
  10. 买点 — SAR 持币转持股 + 低位
  11. 短线 — 5日振幅 RSV 均值
  12. 趋势 — 55日 WRS 的 EMA 趋势

用法:
    result = tdx_signal_catcher(kline)
    result.buy_signals          # 买入信号列表
    result.sell_signals         # 卖出信号列表
    result.综合结论             # 统一结论
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 辅助函数: 通达信专用函数 Python 实现
# ============================================================


def _tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    通达信 SMA(X, N, M) 实现。
    SMA = (M*X + (N-M)*REF(SMA,1)) / N
    对应 pandas: ewm(alpha=2/(N+1), adjust=False).mean() 是 N≥2 时的近似。
    这里用精确递推实现。
    """
    result = np.empty(len(series), dtype=float)
    prev = np.nan
    for i, v in enumerate(series.values):
        if pd.isna(v):
            prev = np.nan
        elif pd.isna(prev):
            prev = v
        else:
            prev = (m * v + (n - m) * prev) / n
        result[i] = prev
    return pd.Series(result, index=series.index)


def _ave_dev(series: pd.Series, period: int) -> pd.Series:
    """
    AVEDEV: 平均绝对偏差 = MEAN(|x - MEAN(x)|)
    """
    return series.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )


def _cross(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    CROSS(S1, S2): S1 上穿 S2 时返回 True。
    """
    prev_diff = (s1 - s2).shift(1)
    curr_diff = (s1 - s2)
    return (prev_diff < 0) & (curr_diff > 0)


# ============================================================
# 核心数据结果类
# ============================================================


@dataclass
class TDXSignalResult:
    """
    提前抄底综合信号结果。
    """
    # 核心指标数值
    多方力度: float = 0.0
    空方力度: float = 0.0
    多方力度1: float = 0.0
    空方力度1: float = 0.0
    短_多方辅助: float = 0.0
    短_空方辅助: float = 0.0
    趋势线: float = 0.0
    牛熊线: float = 0.0
    生命线: float = 0.0

    # 趋势分析字段
    多方力度趋势: str = "震荡"
    空方力度趋势: str = "震荡"
    持股天数: int = 0

    # 买入信号
    buy_signals: List[str] = field(default_factory=list)
    # 卖出信号
    sell_signals: List[str] = field(default_factory=list)
    # 中性信号
    neutral_signals: List[str] = field(default_factory=list)

    # 综合结论
    verdict: str = "中性"
    score: float = 50.0
    recommendation: str = ""
    signals_detail: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "多方力度": round(self.多方力度, 1),
            "空方力度": round(self.空方力度, 1),
            "多方力度趋势": self.多方力度趋势,
            "空方力度趋势": self.空方力度趋势,
            "持股天数": self.持股天数,
            "多方力度1": round(self.多方力度1, 1),
            "空方力度1": round(self.空方力度1, 1),
            "趋势线": round(self.趋势线, 4),
            "牛熊线": round(self.牛熊线, 4),
            "生命线": round(self.生命线, 4),
            "买入信号": self.buy_signals,
            "卖出信号": self.sell_signals,
            "中性信号": self.neutral_signals,
            "综合结论": self.verdict,
            "综合得分": round(self.score, 1),
            "建议": self.recommendation,
            "信号详情": self.signals_detail,
        }


# ============================================================
# 主函数: 提前抄底综合分析
# ============================================================


def tdx_signal_catcher(kline: pd.DataFrame,
                       多线: int = 27, 空线: int = 55) -> TDXSignalResult:
    """
    提前抄底信号综合分析

    Args:
        kline: DataFrame with columns: open, high, low, close, volume[, amount]
        多线: 多方力度计算周期（默认27日）
        空线: 空方力度计算周期（默认55日）

    Returns:
        TDXSignalResult
    """
    result = TDXSignalResult()

    if kline is None or kline.empty or len(kline) < 95:
        result.signals_detail.append("数据不足（需≥95根K线），跳过提前抄底分析")
        return result

    df = kline.copy().reset_index(drop=True)
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']
    prev_close = close.shift(1)
    prev_close_first = prev_close.copy()
    prev_close_first.iloc[0] = close.iloc[0]

    n = len(df)

    # ============================================================
    # 1. 多方力度 vs 空方力度 (27日 WRS 类)
    # ============================================================
    # RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    rsv_multi = (close - low.rolling(多线).min()) / (high.rolling(多线).max() - low.rolling(多线).min()) * 100
    rsv_multi = rsv_multi.fillna(50)

    # KDJ-style: K = SMA(RSV, 5, 1), D = SMA(K, 3, 1), J = 3K - 2D
    k_multi = _tdx_sma(rsv_multi, 5, 1)
    d_multi = _tdx_sma(k_multi, 3, 1)
    j_multi = 3 * k_multi - 2 * d_multi

    # 空方力度 = 100 * (HHV(HIGH, 空线) - CLOSE) / (HHV(HIGH, 空线) - LLV(LOW, 空线))
    hhv_empty = high.rolling(空线).max()
    llv_empty = low.rolling(空线).min()
    空方力度 = 100 * (hhv_empty - close) / (hhv_empty - llv_empty)
    空方力度 = 空方力度.fillna(50)

    # 当前值
    多方力度_curr = j_multi.iloc[-1]
    空方力度_curr = 空方力度.iloc[-1]
    result.多方力度 = 多方力度_curr
    result.空方力度 = 空方力度_curr

    # ============================================================
    # 2. 多方力度1 vs 空方力度1 (24/95日)
    # ============================================================
    rsv_24 = (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min()) * 100
    rsv_24 = rsv_24.fillna(50)
    k_24 = _tdx_sma(rsv_24, 5, 1)
    d_24 = _tdx_sma(k_24, 3, 1)
    j_24 = 3 * k_24 - 2 * d_24

    hhv_95 = high.rolling(95).max()
    llv_95 = low.rolling(95).min()
    空方力度1 = 100 * (hhv_95 - close) / (hhv_95 - llv_95)
    空方力度1 = 空方力度1.fillna(50)

    result.多方力度1 = j_24.iloc[-1]
    result.空方力度1 = 空方力度1.iloc[-1]

    # ============================================================
    # 3. 趋势线/牛熊线/生命线
    # ============================================================
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma15 = close.rolling(15).mean()
    ma20 = close.rolling(20).mean()
    趋势线 = (ma5 + ma10 + ma15 + ma20) / 4
    牛熊线 = close.rolling(26).mean()
    生命线 = close.ewm(span=55, adjust=False).mean()

    result.趋势线 = 趋势线.iloc[-1]
    result.牛熊线 = 牛熊线.iloc[-1]
    result.生命线 = 生命线.iloc[-1]

    JD1 = pd.concat([趋势线, 牛熊线, 生命线], axis=1).max(axis=1)
    JX1 = pd.concat([趋势线, 牛熊线, 生命线], axis=1).min(axis=1)

    # ============================================================
    # 4. 量比 = VOL / REF(VOL, 1)
    # ============================================================
    量比 = volume / volume.shift(1)
    量比 = 量比.fillna(1.0)

    # 一阳定势信号需要 REF(C,1)
    涨 = close / prev_close_first

    # ============================================================
    # 5. SAR (Parabolic SAR)
    # ============================================================
    try:
        import talib
        sar = talib.SAR(high.values, low.values, acceleration=0.02, maximum=0.2)
        sar = pd.Series(sar, index=df.index)
        持币 = (sar >= high).astype(float).replace(0, np.nan) * sar
        持股 = (sar <= low).astype(float).replace(0, np.nan) * sar
    except Exception:
        # fallback: 简单 SAR 近似
        sar = pd.Series(np.nan, index=df.index)
        持币 = pd.Series(np.nan, index=df.index)
        持股 = pd.Series(np.nan, index=df.index)

    # ============================================================
    # 6. 多方辅助 / 空方辅助 (短周期 RSV)
    # ============================================================
    rsv_25 = (close - low.rolling(10).min()) / (high.rolling(25).max() - low.rolling(10).min()) * 100
    rsv_25 = rsv_25.fillna(50)
    # 红线: EMA((CLOSE - LLV(LOW,10)) / (HHV(HIGH,25) - LLV(LOW,10)) * 4, 4) * 25
    rsv_10_25 = (close - low.rolling(10).min()) / (high.rolling(25).max() - low.rolling(10).min()) * 4
    红线 = rsv_10_25.ewm(span=4, adjust=False).mean() * 25

    # 绿线: SMA(RSV10, 3, 1) where RSV10 = (HHV(HIGH,11)-CLOSE)/(HHV(HIGH,11)-LLV(LOW,11))*100
    rsv10 = (high.rolling(11).max() - close) / (high.rolling(11).max() - low.rolling(11).min()) * 100
    rsv10 = rsv10.fillna(50)
    绿线 = _tdx_sma(rsv10, 3, 1)

    # 多方/空方辅助: 多方时取红线/绿线
    多方辅助 = np.where(空方力度1 < j_24, 红线, np.nan)
    空方辅助 = np.where(空方力度1 > j_24, 绿线, np.nan)
    多方辅助 = pd.Series(多方辅助, index=df.index)
    空方辅助 = pd.Series(空方辅助, index=df.index)

    result.短_多方辅助 = 多方辅助.iloc[-1] if not pd.isna(多方辅助.iloc[-1]) else 0
    result.短_空方辅助 = 空方辅助.iloc[-1] if not pd.isna(空方辅助.iloc[-1]) else 0

    # ============================================================
    # 7. 判断各信号
    # ============================================================
    curr_close = close.iloc[-1]
    prev_c = prev_close_first.iloc[-1]
    curr_open = open_.iloc[-1]
    curr_high = high.iloc[-1]
    curr_low = low.iloc[-1]
    curr_vol = volume.iloc[-1]
    curr_vol_ratio = 量比.iloc[-1]
    curr_多力 = 多方力度_curr
    curr_空力 = 空方力度_curr
    curr_多力1 = result.多方力度1
    curr_空力1 = result.空方力度1
    curr_趋势 = 趋势线.iloc[-1]
    curr_牛熊 = 牛熊线.iloc[-1]
    curr_生命 = 生命线.iloc[-1]
    curr_JD1 = JD1.iloc[-1]
    curr_JX1 = JX1.iloc[-1]

    detail_lines = []

    # --- 强弱对比 ---
    if 多方力度_curr > 空方力度_curr:
        strength = "强"
    else:
        strength = "弱"
    detail_lines.append(f"多方力度={多方力度_curr:.1f} vs 空方力度={空方力度_curr:.1f} → {strength}")

    # --- 信号 1: 买点 (SAR 持币转持股) ---
    if len(sar) >= 2:
        prev_sar = sar.iloc[-2]
        curr_sar = sar.iloc[-1]
        prev_hold_cash = bool(prev_sar >= high.iloc[-2]) if not pd.isna(prev_sar) else False
        curr_hold_stock = bool(curr_sar <= low.iloc[-1]) if not pd.isna(curr_sar) else False

        if prev_hold_cash and curr_hold_stock and 多方力度_curr <= 50:
            result.buy_signals.append("买点(SAR持币转持股+低位)")
            detail_lines.append("📌 买点触发：SAR由持币转持股，多方力度≤50")

    # --- 信号 2: 一阳定势 ---
    cross_multi_empty = _cross(j_multi, 空方力度).iloc[-1]
    if (curr_close > curr_JD1 and
            curr_open < curr_JX1 and
            curr_vol_ratio > 1.0 and
            cross_multi_empty and
            curr_close / prev_c > 1.02):
        result.buy_signals.append("一阳定势(一阳包多均线+放量金叉)")
        detail_lines.append("🔥 一阳定势：一阳包多均线 + 放量 + 多方金叉")

    # --- 信号 3: 红钻石 ---
    if (curr_close / prev_c > 1.03 and
            curr_vol_ratio > 1.1 and
            cross_multi_empty):
        result.buy_signals.append("红钻石(大阳线+放量金叉)")
        detail_lines.append("💎 红钻石：涨幅>3% + 量比>1.1 + 多方金叉")

    # --- 信号 4: 金银袋 ---
    if (curr_close > curr_生命 and
            curr_low < curr_生命 and
            curr_close > curr_趋势 and
            cross_multi_empty and
            curr_close / prev_c > 1.02 and
            curr_vol_ratio > 1.1 and
            curr_close > curr_open and
            curr_high / max(curr_open, curr_close) < 1.05):
        result.buy_signals.append("金银袋(回测均线不破+多方金叉)")
        detail_lines.append("🥇 金银袋：回测生命线不破 + 多方金叉 + 量价配合")

    # --- 信号 5: 龙抬头 (KDJ金叉 + 趋势安全区 + 涨幅<10%) ---
    # 使用27日KDJ的金叉
    a6 = k_multi
    a7 = d_multi
    a65 = _cross(a6, a7).iloc[-1]
    a17 = close.rolling(90).mean()
    a26 = (close - a17) / a17 * 100
    curr_a26 = a26.iloc[-1]
    # 安全区: 趋势线2 < 0, 其中趋势线2 = MA((CLOSE-MA40)/MA40*100, 2)
    ma40 = close.rolling(40).mean()
    趋势线2 = ((close - ma40) / ma40 * 100).rolling(2).mean()
    curr_趋势线2 = 趋势线2.iloc[-1]
    安全 = curr_趋势线2 < -10 if not pd.isna(curr_趋势线2) else False

    if (a65 and 安全 and
            (a6.iloc[-1] - a7.iloc[-1]) > 1.5 and
            curr_open != curr_high and
            curr_a26 < -10):
        result.buy_signals.append("龙抬头(KDJ金叉+趋势安全区)")
        detail_lines.append("🐉 龙抬头：KDJ金叉 + 趋势安全区 + 涨幅受限")
    elif 安全 and _cross(a6, a7).iloc[-1]:
        result.buy_signals.append("龙抬头(KDJ安全区金叉)")
        detail_lines.append("🐉 龙抬头：KDJ在安全区金叉")

    # --- 信号 6: 钻石底 ---
    if not pd.isna(curr_趋势线2) and curr_趋势线2 < -20:
        result.buy_signals.append("钻石底(价格深跌偏离40日均线)")
        detail_lines.append("💠 钻石底：价格偏离40日均线超过-20%，深度超跌")

    # --- 信号 7: 抄底 (WR双周期极端超卖) ---
    # VARL2 = (TYP - MA(TYP,14)) / (0.015 * AVEDEV(TYP,14))
    typ = (high + low + close) / 3
    varl2 = (typ - typ.rolling(14).mean()) / (0.015 * _ave_dev(typ, 14))
    varl3 = (typ - typ.rolling(70).mean()) / (0.015 * _ave_dev(typ, 70))
    varl5 = np.where(
        (varl2 >= 150) & (varl2 < 200) & (varl3 >= 150) & (varl3 < 200), 10,
        np.where((varl2 <= -150) & (varl2 > -200) & (varl3 <= -150) & (varl3 > -200), -10, 0)
    )
    # PJX = MA(100*(C-LLV(C,34))/(HHV(H,34)-LLV(L,34)), 5) - 20
    pjx = (100 * (close - close.rolling(34).min()) / (high.rolling(34).max() - low.rolling(34).min())).rolling(5).mean() - 20
    varl1b = np.where(
        (varl3 >= 200) & (varl2 >= 150), 15,
        np.where((varl3 <= -200) & (varl2 <= -150), -15, varl5)
    )
    varl1b = pd.Series(varl1b, index=df.index)

    curr_varl1b = varl1b.iloc[-1]
    prev_varl1b = varl1b.iloc[-2] if len(df) >= 2 else 0
    if prev_varl1b == -15 and curr_varl1b > -15:
        result.buy_signals.append("抄底(WR双周期超卖反弹)")
        detail_lines.append("⭐ 抄底信号：WR双周期极端超卖后转强")

    # --- 信号 8: 逃顶 (WR双周期极端超买) ---
    var1b = np.where(
        (varl3 >= 200) & (varl2 >= 150), 15,
        np.where((varl3 <= -200) & (varl2 <= -150), -15, varl5)
    ) + 60
    var1b = pd.Series(var1b, index=df.index)
    # FILTER(VAR1B=75, 5) - 至少5个bar满足75
    filtered_75 = (var1b == 75).rolling(5).sum() >= 1
    if filtered_75.iloc[-1]:
        result.sell_signals.append("逃顶(WR双周期超买)")
        detail_lines.append("🚩 逃顶信号：WR双周期极端超买")

    # --- 信号 9: 主力吸筹 ---
    # VAR5555 = LLV(LOW, 30)
    var5555 = low.rolling(30).min()
    # VAR3333 = SMA(ABS(LOW - REF(LOW,1)), 3, 1) / SMA(MAX(LOW - REF(LOW,1), 0), 3, 1) * 100
    # 这实际上是类似 KDJ 的 RSV 计算，使用 LOW
    low_delta = low - low.shift(1)
    abs_low_delta = low_delta.abs()
    max_low_delta = np.maximum(low_delta, 0)
    rsv_low = abs_low_delta.rolling(3).mean() / max_low_delta.rolling(3).mean() * 100
    rsv_low = rsv_low.fillna(100)
    # VAR4444 = EMA(IF(CLOSE*1.3, RSV*10, RSV/10), 3)
    # CLOSE*1.3 总是 True（价格>0），所以实际是 RSV*10
    var4444 = (rsv_low * 10).ewm(span=3, adjust=False).mean()
    var6666 = var4444.rolling(30).max()
    var7777 = (close.rolling(58).mean() > 0).astype(int)
    # 主力吸筹 = EMA(IF(LOW<=VAR5555, (VAR4444+VAR6666*2)/2, 0), 3) / 618 * VAR7777
    cond = low <= var5555
    raw = np.where(cond, (var4444 + var6666 * 2) / 2, 0)
    主力吸筹 = pd.Series(raw).ewm(span=3, adjust=False).mean() / 618 * var7777
    主力吸筹_val = 主力吸筹.iloc[-1] if not pd.isna(主力吸筹.iloc[-1]) else 0

    if 主力吸筹_val > 50:
        result.buy_signals.append(f"主力吸筹({主力吸筹_val:.0f})")
        detail_lines.append(f"📊 主力吸筹：{主力吸筹_val:.0f}，底部吸筹明显")
    elif 主力吸筹_val > 20:
        detail_lines.append(f"📊 主力吸筹：{主力吸筹_val:.0f}，温和吸筹")

    # --- 信号 10: 强势持股 (多方>空方且多方<空方辅助) ---
    if (curr_多力 > curr_空力 and
            curr_多力 < curr_空力1 and
            curr_多力 < result.短_多方辅助 if result.短_多方辅助 > 0 else False):
        result.buy_signals.append("强势持股(多方>空方)")
        detail_lines.append("💪 强势持股信号")

    # ============================================================
    # 状态类信号（非当日触发，而是当前状态描述）
    # ============================================================

    # 已处于多头强势区（多方>空方且站上三均线最高）
    if (curr_多力 > curr_空力 and
            curr_close > curr_JD1 and
            curr_close > curr_趋势 and
            curr_close > curr_生命):
        result.buy_signals.append("强势多头(多方>空方+站上三线)")
        detail_lines.append("🚀 强势多头：多方>空方 + 站上趋势线+生命线")

    # 超跌反弹机会（多方>空方且趋势线2<-15但不<-20）
    if (curr_多力 > curr_空力 and
            not pd.isna(curr_趋势线2) and
            curr_趋势线2 > -20 and curr_趋势线2 < -15):
        result.buy_signals.append("超跌反弹(多方>空方+深度回调)")
        detail_lines.append("📈 超跌反弹：多方占优 + 价格偏离40日均线-15%~-20%")

    # 多方占优但价格破位（多方>空方但价格跌破三均线）
    if (curr_多力 > curr_空力 and
            (curr_close < curr_JD1 or curr_close < curr_生命)):
        result.neutral_signals.append("多方占优但价格破位")
        detail_lines.append("⚠️ 多方>空方，但价格跌破关键均线，趋势有分歧")

    # 底部震荡（多方接近空方且价格在低位）
    if (curr_多力 < 60 and curr_空力 < 60 and
            abs(curr_多力 - curr_空力) < 10 and
            not pd.isna(curr_趋势线2) and curr_趋势线2 < -5):
        result.neutral_signals.append("底部震荡(多空均衡+低位)")
        detail_lines.append("⚖️ 底部震荡：多空均衡 + 价格在低位区间")

    # 高位滞涨（多方>空方但趋势线2接近+10以上，且量比<1）
    if (curr_多力 > curr_空力 and
            curr_多力 > 70 and
            not pd.isna(curr_趋势线2) and curr_趋势线2 > 8 and
            curr_vol_ratio < 1.0):
        result.sell_signals.append("高位滞涨(多方高位+缩量)")
        detail_lines.append("⚠️ 高位滞涨：多方高位 + 趋势过强 + 缩量")

    # 空方主导（空方>多方且空方>70）—— 已移至趋势分析之后

    # --- 信号 11: 短线 ---
    # A4 = EMA((P-A2)/(A3), 2) * 100, P=(2C+H+L+O)/5
    p = (2 * close + high + low + open_) / 5
    a1 = p.rolling(15).max()
    a2 = p.rolling(15).min()
    a3 = a1 - a2
    a4 = ((p - a2) / a3).ewm(span=2, adjust=False).mean() * 100
    a4 = a4.fillna(50)

    # --- 信号 12: 趋势 (55日WRS的EMA趋势) ---
    rsv_55 = (close - low.rolling(55).min()) / (high.rolling(55).max() - low.rolling(55).min()) * 100
    rsv_55 = rsv_55.fillna(50)
    k_55 = _tdx_sma(rsv_55, 5, 1)
    d_55 = _tdx_sma(k_55, 3, 1)
    j_55 = 3 * k_55 - 2 * d_55
    趋势 = j_55.ewm(span=3, adjust=False).mean() - 10

    curr_趋势 = 趋势.iloc[-1]
    if 趋势.iloc[-2] <= 13 and curr_趋势 > 13:
        detail_lines.append("📈 趋势转多：55日WRS趋势线突破13")
    elif 趋势.iloc[-2] >= 90 and curr_趋势 < 90:
        detail_lines.append("📉 趋势转空：55日WRS趋势线跌破90")
    else:
        if curr_趋势 > 50:
            detail_lines.append(f"📈 趋势偏多：趋势线={curr_趋势:.1f}")
        else:
            detail_lines.append(f"📉 趋势偏空：趋势线={curr_趋势:.1f}")

    # ============================================================
    # 趋势分析: 多方/空方力度多周期趋势 + 持股天数
    # ============================================================
    多力_series = j_multi if hasattr(j_multi, 'iloc') else j_multi
    空力_series = 空方力度 if hasattr(空方力度, 'iloc') else 空方力度

    # 多周期趋势判断: 比较最近N日的均值变化
    def _calc_trend(series, periods=(5, 10, 20)):
        """计算多周期趋势,返回(near,mid,far,direction,detail)"""
        n = len(series)
        near = series.iloc[-1] if n >= 1 else 50
        near_mid = series.iloc[-min(5,n)//2] if n >= 5 else 50
        mid_mid = series.iloc[-min(10,n)//2] if n >= 10 else 50
        far_mid = series.iloc[-min(20,n)//2] if n >= 20 else 50
        # 趋势方向: 近>中>远=持续上升, 反之下跌
        if near > near_mid + 5 and near_mid > mid_mid + 5:
            direction = "持续上升"
        elif near < near_mid - 5 and near_mid < mid_mid - 5:
            direction = "持续下降"
        elif near > near_mid + 5:
            direction = "近期上升"
        elif near < near_mid - 5:
            direction = "近期下降"
        elif near > far_mid + 10:
            direction = "中期上升"
        elif near < far_mid - 10:
            direction = "中期下降"
        else:
            direction = "震荡"
        detail = f"近期={near:.1f}(5d前={near_mid:.1f}),中期={mid_mid:.1f}(10d前),远期={far_mid:.1f}(20d前)"
        return near, near_mid, mid_mid, far_mid, direction, detail

    多力_near, 多力_5d, 多力_10d, 多力_20d, 多力_trend, 多力_detail = _calc_trend(多力_series)
    空力_near, 空力_5d, 空力_10d, 空力_20d, 空力_trend, 空力_detail = _calc_trend(空力_series)

    # 持股天数: 连续多方>空方天数
    diff_series = 多力_series - 空力_series
    consecutive_hold = 0
    for i in range(len(diff_series) - 1, -1, -1):
        if diff_series.iloc[i] > 0:
            consecutive_hold += 1
        else:
            break

    # 记录趋势到结果
    result.多方力度趋势 = 多力_trend
    result.空方力度趋势 = 空力_trend
    result.持股天数 = consecutive_hold

    # 添加趋势信号
    if 多力_trend == "持续上升" and 空力_trend in ("持续下降", "近期下降", "中期下降"):
        result.buy_signals.append(f"多空趋势改善(多方{多力_trend}+空方{空力_trend})")
        detail_lines.append(f"📈 多空趋势改善：多方力度{多力_trend}({多力_detail}) + 空方力度{空力_trend}({空力_detail})")
    elif 多力_trend in ("持续上升", "近期上升", "中期上升") and 空力_trend in ("持续下降", "近期下降", "中期下降"):
        result.buy_signals.append(f"多空背离转好(多方升+空方降)")
        detail_lines.append(f"📈 多空背离转好：多方{多力_trend} + 空方{空力_trend}")
    elif 多力_trend == "持续上升" and consecutive_hold >= 5:
        result.buy_signals.append(f"持续持股({consecutive_hold}日连续多方占优)")
        detail_lines.append(f"💪 持续持股：连续{consecutive_hold}日多方占优，趋势稳定")
    elif 空力_trend in ("持续上升", "近期上升") and 多力_trend in ("持续下降", "近期下降"):
        result.sell_signals.append(f"多空趋势恶化(多方降+空方升)")
        detail_lines.append(f"📉 多空趋势恶化：多方{多力_trend} + 空方{空力_trend}")
    elif consecutive_hold >= 3 and 多力_trend == "震荡":
        result.neutral_signals.append(f"持股{consecutive_hold}日但多空震荡")
        detail_lines.append(f"⚠️ 持续持股{consecutive_hold}日，但多空力度震荡未明确方向")

    # 趋势详情
    detail_lines.append(f"📊 多方力度趋势: {多力_trend} | {多力_detail}")
    detail_lines.append(f"📊 空方力度趋势: {空力_trend} | {空力_detail}")
    detail_lines.append(f"📊 持股天数: {consecutive_hold}日连续多方占优")

    # 空方主导（空方>多方且空方>70）—— 趋势分析后执行
    if (curr_空力 > curr_多力 and curr_空力 > 70):
        if 空力_trend in ("持续下降", "近期下降", "中期下降") and 多力_trend in ("持续上升", "近期上升", "中期上升"):
            result.neutral_signals.append(f"空方强势但趋势改善(空方{空力_trend}+多方{多力_trend})")
            detail_lines.append(f"⚠️ 空方力度>70，但趋势正在改善（空方{空力_trend}+多方{多力_trend}），关注后续确认")
        else:
            result.sell_signals.append("空方主导(空方力度>70)")
            detail_lines.append("📉 空方主导：空方力度>70，强势下跌趋势")

    # ============================================================
    # 综合评分
    # ============================================================
    score = 50

    # 强弱对比(基础)
    if curr_多力 > curr_空力:
        score += 10
    else:
        score -= 10
    if curr_多力1 > curr_空力1:
        score += 5
    else:
        score -= 5

    # 趋势加分(替代纯静态判断)
    if 多力_trend == "持续上升":
        score += 15
    elif 多力_trend in ("近期上升", "中期上升"):
        score += 10
    elif 多力_trend == "持续下降":
        score -= 15
    elif 多力_trend in ("近期下降", "中期下降"):
        score -= 10

    if 空力_trend == "持续下降":
        score += 10
    elif 空力_trend in ("近期下降", "中期下降"):
        score += 5
    elif 空力_trend == "持续上升":
        score -= 10
    elif 空力_trend in ("近期上升", "中期上升"):
        score -= 5

    # 持股天数加分
    if consecutive_hold >= 10:
        score += 10
    elif consecutive_hold >= 5:
        score += 5
    elif consecutive_hold >= 3:
        score += 3

    # 均线位置
    if curr_close > curr_JD1:
        score += 10
    elif curr_close < curr_JX1:
        score -= 10

    # 买入信号加分
    for sig in result.buy_signals:
        if "钻石底" in sig or "抄底" in sig:
            score += 15
        elif "一阳定势" in sig or "红钻石" in sig or "金银袋" in sig:
            score += 20
        elif "龙抬头" in sig:
            score += 15
        elif "主力吸筹" in sig:
            score += 10
        elif "买点" in sig:
            score += 10
        elif "趋势改善" in sig or "背离转好" in sig or "持续持股" in sig:
            score += 8
        else:
            score += 5

    # 卖出信号扣分
    for sig in result.sell_signals:
        score -= 15

    score = max(0, min(100, score))
    result.score = score

    # ============================================================
    # 综合结论
    # ============================================================
    buy_count = len(result.buy_signals)
    sell_count = len(result.sell_signals)

    if buy_count >= 3 or (buy_count >= 1 and score >= 75):
        result.verdict = "强烈看多"
    elif buy_count >= 2 or score >= 65:
        result.verdict = "看多"
    elif sell_count >= 1 or score < 40:
        result.verdict = "看空"
    elif buy_count == 1 or score >= 55:
        result.verdict = "偏多"
    else:
        result.verdict = "中性"

    if result.verdict == "强烈看多":
        result.recommendation = f"强烈看多({score}分)，买入信号{buy_count}个，建议积极布局"
    elif result.verdict == "看多":
        if buy_count > 0:
            result.recommendation = f"看多({score}分)，有{buy_count}个买入信号，可择机入场"
        else:
            result.recommendation = f"看多({score}分)，多方力量占优，暂无明确买点信号，可关注后续"
    elif result.verdict == "偏多":
        result.recommendation = f"偏多({score}分)，有{buy_count}个买入信号，可轻仓尝试"
    elif result.verdict == "看空":
        result.recommendation = f"看空({score}分)，卖出信号{sell_count}个，建议观望或减仓"
    else:
        result.recommendation = f"中性({score}分)，暂无明确信号，建议观望"

    result.signals_detail = detail_lines
    return result