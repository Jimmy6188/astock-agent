"""
A股六维度智能分析引擎
技术面 / 估值 / 资金面 / 基本面 / 财报质量 / 舆情情绪
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
try:
    from .data_provider import AStockDataProvider
except ImportError:
    from data_provider import AStockDataProvider

logger = logging.getLogger(__name__)

# ==================== 评分阈值常量 ====================
# 估值维度
PE_LOW = 15          # PE低于此值视为低估
PE_REASONABLE = 25   # PE低于此值为合理
PE_HIGH = 40         # PE高于此值为高估
PB_LOW = 1.5         # PB低于此值为破净
PB_HIGH = 6          # PB高于此值为高估
PEG_UNDERVALUED = 0.8  # PEG低于此值为低估
PEG_OVERVALUED = 2.0   # PEG高于此值为高估

# 基本面维度
ROE_EXCELLENT = 20   # ROE优秀阈值
ROE_GOOD = 12        # ROE良好阈值
ROE_POOR = 6         # ROE偏弱阈值
REVENUE_HIGH_GROWTH = 30   # 营收高增长阈值
REVENUE_MED_GROWTH = 15    # 营收稳健增长阈值
DEBT_RATIO_HIGH = 70       # 负债率高阈值
DEBT_RATIO_LOW = 30        # 负债率低阈值
CURRENT_RATIO_LOW = 1.0    # 流动比率低阈值

# 技术面维度
RSI_OVERBOUGHT = 70   # RSI超买
RSI_OVERSOLD = 30     # RSI超卖
KDJ_LOW = 20          # KDJ低位金叉
KDJ_HIGH = 80         # KDJ高位死叉
VOLUME_RATIO_HIGH = 2.0  # 放量阈值
PCT_5D_OVERBOUGHT = 10  # 5日涨幅超买
PCT_5D_OVERSOLD = -10   # 5日跌幅超卖
BB_LOWER = 10         # 布林带下轨位置
BB_UPPER = 90         # 布林带上轨位置
TURN_OVER_HEAVY = 10   # 换手率活跃
TURN_OVER_SPECULATIVE = 20  # 换手率投机

# 财报质量
OCF_EPS_MATCH_HIGH = 1.0
OCF_EPS_MATCH_MED = 0.5
EPS_LOW = 0.1         # EPS偏低阈值


class DimensionScore:
    """单维度评分结果"""
    def __init__(self, name: str, score: float, max_score: float = 100,
                 rating: str = "", details: dict = None, signals: list = None):
        self.name = name
        self.score = min(max(score, 0), max_score)
        self.max_score = max_score
        self.rating = rating or _score_to_rating(self.score)
        self.details = details or {}
        self.signals = signals or []

    def to_dict(self) -> dict:
        return {
            "dimension": self.name,
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "rating": self.rating,
            "details": self.details,
            "signals": self.signals,
        }


def _score_to_rating(score: float) -> str:
    if score >= 80: return "强烈看好"
    if score >= 65: return "看好"
    if score >= 50: return "中性偏多"
    if score >= 35: return "中性偏空"
    if score >= 20: return "看空"
    return "强烈看空"


class TechnicalAnalyzer:
    """维度一：技术面分析 - 均线系统、MACD、RSI、KDJ、布林带、成交量"""

    @staticmethod
    def analyze(kline_df: pd.DataFrame) -> DimensionScore:
        if kline_df is None or kline_df.empty or len(kline_df) < 30:
            return DimensionScore("技术面", 40, details={"error": "K线数据不足"})

        details = {}
        signals = []
        score = 50.0  # 基准分

        try:
            close = kline_df["close"].astype(float)
            high = kline_df["high"].astype(float)
            low = kline_df["low"].astype(float)
            volume = kline_df["volume"].astype(float)
            n = len(close)

            # --- 均线系统 ---
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean() if n >= 60 else pd.Series()

            last_close = close.iloc[-1]
            last_ma5 = ma5.iloc[-1]
            last_ma10 = ma10.iloc[-1]
            last_ma20 = ma20.iloc[-1]

            details["price"] = round(last_close, 2)
            details["ma5"] = round(last_ma5, 2) if pd.notna(last_ma5) else None
            details["ma10"] = round(last_ma10, 2) if pd.notna(last_ma10) else None
            details["ma20"] = round(last_ma20, 2) if pd.notna(last_ma20) else None

            # 多头排列加分
            if pd.notna(last_ma5) and pd.notna(last_ma10) and pd.notna(last_ma20):
                if last_ma5 > last_ma10 > last_ma20 and last_close > last_ma5:
                    score += 15
                    signals.append("均线多头排列，趋势向上")
                elif last_ma5 < last_ma10 < last_ma20 and last_close < last_ma5:
                    score -= 15
                    signals.append("均线空头排列，趋势向下")
                    signals.append("均线空头排列，趋势向下")

            # 价格站上关键均线
            if pd.notna(last_ma20):
                if last_close > last_ma20:
                    score += 5
                else:
                    score -= 5

            # ============================
            # MACD 增强分析（基于通达信副图指标逻辑）
            # 包含：趋势4档评定、低位/二次/空中加油金叉、
            # 底背离/顶背离、KP13吸拉派落周期
            # ============================
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            macd_bar = 2 * (dif - dea)  # MACD柱

            ld = dif.iloc[-1]
            le = dea.iloc[-1]
            lm = macd_bar.iloc[-1]
            pd_ = dif.iloc[-2] if n >= 2 else ld
            pe_ = dea.iloc[-2] if n >= 2 else le
            pm_ = macd_bar.iloc[-2] if n >= 2 else lm

            details["macd_dif"] = round(ld, 4) if pd.notna(ld) else None
            details["macd_dea"] = round(le, 4) if pd.notna(le) else None
            details["macd_hist"] = round(lm, 4) if pd.notna(lm) else None

            if pd.notna(ld) and pd.notna(le) and pd.notna(pm_):
                # --- 1. 趋势4档评定 ---
                # 参考副图: COUNT(MACD>0,5)>=3 AND MACD>=REF(MACD,1)
                pos_cnt = (macd_bar.iloc[-5:] > 0).sum() if n >= 5 else 0
                neg_cnt = (macd_bar.iloc[-5:] < 0).sum() if n >= 5 else 0
                macd_rising = lm >= pm_
                macd_trend = "趋势不明"
                macd_trend_score = 0
                if n >= 5 and pos_cnt >= 3:
                    if macd_rising:
                        macd_trend = "多头势强"
                        macd_trend_score = 14
                        signals.append("MACD多头势强(柱体≥3/5日正且柱体放大)，持股待涨")
                    else:
                        macd_trend = "多头转弱"
                        macd_trend_score = -4
                        signals.append("MACD多头转弱(柱体缩量)，逢高减仓")
                elif n >= 5 and neg_cnt >= 3:
                    if macd_rising:
                        # MACD柱由负趋零 = 空头动能衰减 = 利好
                        macd_trend = "空头转弱"
                        macd_trend_score = 8
                        signals.append("MACD空头转弱(柱体萎缩)，择机入场")
                    else:
                        # MACD柱由零趋负 = 空头动能增强 = 利空
                        macd_trend = "空头势强"
                        macd_trend_score = -16
                        signals.append("MACD空头势强(柱体≥3/5日负且柱体扩大)，清仓离场")
                else:
                    macd_trend = "趋势不明"
                    macd_trend_score = 0
                    signals.append("MACD趋势不明，持币观望")
                score += macd_trend_score
                details["macd_trend"] = macd_trend

                # --- 2. CROSS 检测 (CROSS(A,B) = REF(A,1)<=REF(B,1) AND A>B) ---
                cross_dif_dea = (pd_ <= pe_) and (ld > le)
                cross_dea_dif = (pe_ <= pd_) and (le > ld)

                # --- 3. 低位金叉: CROSS(DIF,DEA) AND DIF < -0.1 ---
                if cross_dif_dea and ld < -0.1:
                    score += 18
                    signals.append("MACD【低位金叉】DIF在零轴下方金叉，反弹概率大")
                    details["macd_pattern"] = "低位金叉"

                # --- 4. 二次金叉: 21日内第二次CROSS(DIF,DEA) AND DEA<0 ---
                if cross_dif_dea and le < 0 and n >= 22:
                    cross_cnt = 0
                    for i in range(-21, 0):
                        if i < -len(dif) or i-1 < -len(dif):
                            continue
                        idx = i
                        if idx - 1 >= 0:
                            if dif.iloc[idx - 1] <= dea.iloc[idx - 1] and dif.iloc[idx] > dea.iloc[idx]:
                                cross_cnt += 1
                    if cross_cnt >= 1:  # 加上本次共2次
                        score += 20
                        signals.append("MACD【二次金叉】21日内第二次金叉，多头信号强")
                        details["macd_pattern"] = "二次金叉"

                # --- 5. 空中加油: DIF上翘后再下探但不下穿DEA再上翘 ---
                # DIF>REF(DIF,1) AND DIF>=DEA AND REF(DIF,1)<=REF(DIF,2) AND REF(DIF,1)>=REF(DEA,1)
                if n >= 4:
                    if ld > pd_ and ld >= le and pd_ <= dif.iloc[-3] and pd_ >= dea.iloc[-3]:
                        score += 16
                        signals.append("MACD【空中加油】DIF短暂回调后再度上翘，加速上涨信号")
                        details["macd_pattern"] = "空中加油"

                # --- 6. 底背离: 价格新低但DIF未新低(在零轴下方) ---
                if n >= 25 and le < 0:
                    # 找近25日内的前一个DIF低点
                    dif_vals = dif.iloc[-25:].values
                    close_vals = close.iloc[-25:].values
                    # 取除最近3日外的DIF最小值
                    if len(dif_vals) >= 5:
                        prev_dif_min = np.min(dif_vals[:-3])
                        prev_close_min = np.min(close_vals[:-3])
                        cur_dif_min = np.min(dif_vals[-3:])
                        cur_close_min = np.min(close_vals[-3:])
                        if cur_close_min < prev_close_min and cur_dif_min > prev_dif_min:
                            score += 15
                            signals.append("MACD【底背离】价格创新低但DIF走高，底部反转信号")
                            details["macd_pattern"] = "底背离"

                # --- 7. 顶背离: 价格新高但DIF未新高 ---
                if n >= 25 and le > 0:
                    dif_vals = dif.iloc[-25:].values
                    close_vals = close.iloc[-25:].values
                    if len(dif_vals) >= 5:
                        prev_dif_max = np.max(dif_vals[:-3])
                        prev_close_max = np.max(close_vals[:-3])
                        cur_dif_max = np.max(dif_vals[-3:])
                        cur_close_max = np.max(close_vals[-3:])
                        if cur_close_max > prev_close_max and cur_dif_max < prev_dif_max:
                            score -= 15
                            signals.append("MACD【顶背离】价格创新高但DIF走低，顶部反转信号")
                            details["macd_pattern"] = "顶背离"

                # --- 8. 柱背离: 5日内MACD柱高点比前次低但价格新高 ---
                if n >= 10:
                    macd_5d_high = macd_bar.iloc[-5:].max()
                    macd_prev_5d_high = macd_bar.iloc[-10:-5].max()
                    close_5d_high = close.iloc[-5:].max()
                    close_prev_5d_high = close.iloc[-10:-5].max()
                    if close_5d_high > close_prev_5d_high and macd_5d_high < macd_prev_5d_high:
                        score -= 8
                        signals.append("MACD【柱顶背离】价格新高但红柱缩短，短期头部")
                    elif close_5d_high < close_prev_5d_high and macd_5d_high > macd_prev_5d_high:
                        score += 5
                        signals.append("MACD【柱底背离】价格新低但绿柱缩短，短期底部")

                # --- 9. KP13 吸拉派落周期 ---
                # VAR1=EMA(EMA(CLOSE,13),13); KP13=(VAR1-REF(VAR1,1))/REF(VAR1,1)*1000
                var1 = close.ewm(span=13).mean().ewm(span=13).mean()
                kp13 = (var1 - var1.shift(1)) / var1.shift(1) * 1000
                kp13_now = kp13.iloc[-1]
                kp13_prev = kp13.iloc[-2] if n >= 2 else kp13_now
                details["kp13"] = round(kp13_now, 2) if pd.notna(kp13_now) else None

                if pd.notna(kp13_now) and pd.notna(kp13_prev):
                    if kp13_now < 0 and kp13_now > kp13_prev:
                        phase = "吸筹"
                        score += 8
                        signals.append(f"KP13【吸筹】KP13={kp13_now:.1f}，主力暗中吸筹")
                    elif kp13_now >= 0 and kp13_now > kp13_prev:
                        phase = "拉升"
                        score += 10
                        signals.append(f"KP13【拉升】KP13={kp13_now:.1f}，主力拉升中")
                    elif kp13_now >= 0 and kp13_now <= kp13_prev:
                        phase = "派发"
                        score -= 8
                        signals.append(f"KP13【派发】KP13={kp13_now:.1f}，主力派发阶段")
                    else:
                        phase = "回落"
                        score -= 10
                        signals.append(f"KP13【回落】KP13={kp13_now:.1f}，股价回落")
                    details["kp13_phase"] = phase

                # --- 10. DIF上翘/下拐标记 ---
                dif_direction = "上翘" if ld > pd_ else "下拐"
                details["macd_dif_dir"] = dif_direction

                # --- 11. 金叉共振 (MACD金叉 + KDJ金叉) ---
                # KDJ已在下方计算，这里需要用到last_k和last_d
                rsv_raw = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min()).replace(0, np.nan) * 100
                k_series = rsv_raw.ewm(com=2).mean()
                d_series = k_series.ewm(com=2).mean()
                j_series = 3 * k_series - 2 * d_series
                lk = k_series.iloc[-1]
                ld_kdj = d_series.iloc[-1]
                pk = k_series.iloc[-2] if n >= 2 else lk
                pd_kdj = d_series.iloc[-2] if n >= 2 else ld_kdj
                kdj_cross = (pk <= pd_kdj) and (lk > ld_kdj)
                details["macd_kdj_resonance"] = bool(cross_dif_dea and kdj_cross)
                if cross_dif_dea and kdj_cross:
                    score += 12
                    signals.append("MACD【金叉共振】DIF/DEA与KDJ同时金叉，多头共振")

            # --- RSI ---
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            last_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

            details["rsi_14"] = round(last_rsi, 1) if pd.notna(last_rsi) else None

            if pd.notna(last_rsi):
                if last_rsi < RSI_OVERSOLD:
                    score += 12
                    signals.append(f"RSI={last_rsi:.1f}，超卖区，可能反弹")
                elif last_rsi > RSI_OVERBOUGHT:
                    score -= 12
                    signals.append(f"RSI={last_rsi:.1f}，超买区，注意回调")
                elif 45 <= last_rsi <= 60:
                    score += 5  # 强势区间

            # --- KDJ ---
            low_9 = low.rolling(9).min()
            high_9 = high.rolling(9).max()
            rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d

            last_k = k.iloc[-1] if len(k) > 0 else 50
            last_d = d.iloc[-1] if len(d) > 0 else 50
            last_j = j.iloc[-1] if len(j) > 0 else 50

            details["kdj_k"] = round(last_k, 1) if pd.notna(last_k) else None
            details["kdj_d"] = round(last_d, 1) if pd.notna(last_d) else None
            details["kdj_j"] = round(last_j, 1) if pd.notna(last_j) else None

            if pd.notna(last_k) and pd.notna(last_d):
                if last_k > last_d and last_k < KDJ_LOW:
                    score += 8
                    signals.append("KDJ低位金叉")
                elif last_k < last_d and last_k > KDJ_HIGH:
                    score -= 8
                    signals.append("KDJ高位死叉")

            # --- 布林带 ---
            ma_mid = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = ma_mid + 2 * std20
            lower = ma_mid - 2 * std20

            last_upper = upper.iloc[-1] if len(upper) > 0 else None
            last_lower = lower.iloc[-1] if len(lower) > 0 else None

            if pd.notna(last_upper) and pd.notna(last_lower):
                range_size = last_upper - last_lower
                if range_size > 0:
                    bb_pos = (last_close - last_lower) / range_size * 100
                    details["boll_position"] = round(bb_pos, 1)
                    if bb_pos < BB_LOWER:
                        score += 8
                        signals.append("价格接近布林下轨，可能超跌")
                    elif bb_pos > BB_UPPER:
                        score -= 8
                        signals.append("价格接近布林上轨，可能超涨")
                else:
                    details["boll_position"] = 50.0  # 布林带收口，无法计算位置

            # --- 成交量分析 ---
            vol_ma5 = volume.rolling(5).mean()
            last_vol = volume.iloc[-1]
            last_vol_ma = vol_ma5.iloc[-1] if len(vol_ma5) > 0 else last_vol

            if pd.notna(last_vol_ma) and last_vol_ma > 0:
                vol_ratio = last_vol / last_vol_ma
                details["volume_ratio"] = round(vol_ratio, 2)
                if vol_ratio > VOLUME_RATIO_HIGH and last_close > close.iloc[-2]:
                    score += 7
                    signals.append(f"放量上涨(量比{vol_ratio:.1f}x)，资金关注度高")
                elif vol_ratio > VOLUME_RATIO_HIGH and last_close < close.iloc[-2]:
                    score -= 7
                    signals.append(f"放量下跌(量比{vol_ratio:.1f}x)，抛压较重")

            # --- 价格动能 ---
            if n >= 5:
                pct_5d = (last_close / close.iloc[-5] - 1) * 100
                details["pct_5d"] = round(pct_5d, 2)
                if pct_5d > PCT_5D_OVERBOUGHT:
                    score -= 3  # 短期涨幅过大，追高风险
                elif pct_5d < PCT_5D_OVERSOLD:
                    score += 3  # 短期超跌

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"技术面分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("技术面", score, details=details, signals=signals)


class ValuationAnalyzer:
    """维度二：估值分析 - PE/PB分位、PEG、PS、股息率"""

    @staticmethod
    def analyze(quote: dict, valuation: dict, financials: dict) -> DimensionScore:
        details = {}
        signals = []
        score = 50.0

        try:
            pe = quote.get("pe_ttm")
            pb = quote.get("pb")
            total_mv = quote.get("total_mv", 0)

            details["pe_ttm"] = pe
            details["pb"] = pb
            details["total_mv_yi"] = round(total_mv / 1e8, 2) if total_mv else None

            # PE评分
            if pe and pe > 0:
                if pe < PE_LOW:
                    score += 15
                    signals.append(f"PE-TTM={pe:.1f}，估值偏低")
                elif pe < PE_REASONABLE:
                    score += 8
                    signals.append(f"PE-TTM={pe:.1f}，估值合理")
                elif pe < PE_HIGH:
                    score -= 3
                    details["pe_note"] = f"PE-TTM={pe:.1f}，估值偏高"
                else:
                    score -= 12
                    signals.append(f"PE-TTM={pe:.1f}，估值过高")
            elif pe and pe < 0:
                score -= 10
                signals.append("公司亏损，PE为负")

            # PB评分
            if pb and pb > 0:
                if pb < PB_LOW:
                    score += 10
                    signals.append(f"PB={pb:.2f}，破净或低PB")
                elif pb < 3:
                    score += 5
                elif pb > PB_HIGH:
                    score -= 8
                    signals.append(f"PB={pb:.2f}，高PB需高增长支撑")

            # 历史分位
            if valuation and "error" not in valuation:
                pe_pct = valuation.get("pe_percentile_3y")
                pb_pct = valuation.get("pb_percentile_3y")
                details["pe_percentile_3y"] = pe_pct
                details["pb_percentile_3y"] = pb_pct

                if pe_pct is not None:
                    if pe_pct < 20:
                        score += 12
                        signals.append(f"PE处于近3年{pe_pct:.0f}%分位，历史低估区间")
                    elif pe_pct > 80:
                        score -= 12
                        signals.append(f"PE处于近3年{pe_pct:.0f}%分位，历史高估区间")

                if pb_pct is not None:
                    if pb_pct < 20:
                        score += 8
                    elif pb_pct > 80:
                        score -= 8

            # PEG估算
            if pe and financials and "error" not in financials:
                yoy = financials.get("revenue_yoy") or financials.get("net_profit_yoy")
                if yoy and yoy != 0 and pe > 0:
                    peg = pe / abs(yoy)
                    details["peg"] = round(peg, 2)
                    if peg < PEG_UNDERVALUED:
                        score += 10
                        signals.append(f"PEG={peg:.2f}，成长性被低估")
                    elif peg < 1.2:
                        score += 5
                    elif peg > PEG_OVERVALUED:
                        score -= 8
                        signals.append(f"PEG={peg:.2f}，估值与增速不匹配")

            # 股息率（从财报推算）
            if financials and "error" not in financials:
                eps = financials.get("eps")
                if eps and eps > 0 and pe and pe > 0:
                    div_yield_est = 1.0 / pe * 0.3  # 假设30%分红率
                    details["div_yield_est"] = f"{div_yield_est*100:.1f}%"

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"估值分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("估值", score, details=details, signals=signals)


class MoneyFlowAnalyzer:
    """维度三：资金流向分析 - 主力净流入、大单/超大单、龙虎榜"""

    @staticmethod
    def analyze(money_flow: dict, lhb: list, quote: dict) -> DimensionScore:
        details = {}
        signals = []
        score = 50.0

        try:
            if money_flow and "error" not in money_flow:
                main_net = money_flow.get("main_net_inflow", 0)
                main_pct = money_flow.get("main_net_pct", 0)
                super_large = money_flow.get("super_large_net", 0)
                large_net = money_flow.get("large_net", 0)
                small_net = money_flow.get("small_net", 0)

                details["main_net_inflow_wan"] = round(main_net / 10000, 1)
                details["main_net_pct"] = round(main_pct, 2)
                details["super_large_wan"] = round(super_large / 10000, 1)
                details["large_wan"] = round(large_net / 10000, 1)
                details["small_wan"] = round(small_net / 10000, 1)

                # 主力净流入评分
                amount = quote.get("amount", 1)  # 成交额
                if amount and amount > 0:
                    flow_ratio = main_net / amount
                    details["flow_to_amount_ratio"] = round(flow_ratio, 3)

                    if flow_ratio > 0.05:
                        score += 18
                        signals.append(f"主力大幅净流入({flow_ratio*100:.1f}%成交额)，资金做多意愿强")
                    elif flow_ratio > 0.02:
                        score += 10
                        signals.append("主力温和净流入")
                    elif flow_ratio < -0.05:
                        score -= 18
                        signals.append(f"主力大幅净流出({abs(flow_ratio)*100:.1f}%成交额)，资金撤离明显")
                    elif flow_ratio < -0.02:
                        score -= 10
                        signals.append("主力温和净流出")

                # 散户反向指标（小单流入通常意味着散户接盘）
                if small_net > 0 and main_net < 0:
                    score -= 5
                    signals.append("主力流出+散户流入，警惕出货")

                # 超大单动向
                if abs(super_large) > abs(main_net) * 0.5:
                    if super_large > 0:
                        score += 5
                        signals.append("超大单积极买入")
                    else:
                        score -= 5
                        signals.append("超大单积极卖出")
            else:
                details["money_flow_note"] = "暂无资金流数据"

            # 龙虎榜信号
            if lhb:
                details["lhb_count"] = len(lhb)
                recent_lhb = [l for l in lhb if l.get("net_buy", 0) > 0]
                if recent_lhb:
                    total_net_buy = sum(l["net_buy"] for l in recent_lhb)
                    details["lhb_total_net_buy_wan"] = round(total_net_buy / 10000, 1)
                    score += 8
                    signals.append(f"近两月龙虎榜{len(lhb)}次上榜，机构净买入{total_net_buy/10000:.0f}万")
                else:
                    score -= 3
                    signals.append("龙虎榜上榜但以卖出为主")

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"资金面分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("资金面", score, details=details, signals=signals)


class FundamentalAnalyzer:
    """维度四：基本面分析 - 盈利能力、成长性、偿债能力、运营效率"""

    @staticmethod
    def analyze(financials: dict, quote: dict) -> DimensionScore:
        details = {}
        signals = []
        score = 50.0

        try:
            if not financials or "error" in financials:
                return DimensionScore("基本面", 35, details={"error": "无财务数据"}, signals=["缺乏财务数据"])

            # 盈利能力
            roe = financials.get("roe")
            net_margin = financials.get("net_margin")
            gross_margin = financials.get("gross_margin")

            details["roe"] = roe
            details["net_margin"] = net_margin
            details["gross_margin"] = gross_margin

            if roe is not None:
                if roe > ROE_EXCELLENT:
                    score += 15
                    signals.append(f"ROE={roe:.1f}%，盈利能力优秀")
                elif roe > ROE_GOOD:
                    score += 8
                    signals.append(f"ROE={roe:.1f}%，盈利能力良好")
                elif roe > ROE_POOR:
                    score += 2
                elif roe > 0:
                    score -= 5
                    signals.append(f"ROE仅{roe:.1f}%，盈利能力偏弱")
                else:
                    score -= 15
                    signals.append("ROE为负，公司亏损")

            if net_margin is not None:
                if net_margin > 25:
                    score += 8
                elif net_margin < 5 and net_margin > 0:
                    score -= 5

            # 成长性
            rev_yoy = financials.get("revenue_yoy")
            profit_yoy = financials.get("net_profit_yoy")

            details["revenue_yoy"] = rev_yoy
            details["profit_yoy"] = profit_yoy

            if rev_yoy is not None:
                if rev_yoy > REVENUE_HIGH_GROWTH:
                    score += 12
                    signals.append(f"营收同比+{rev_yoy:.1f}%，高速增长")
                elif rev_yoy > REVENUE_MED_GROWTH:
                    score += 6
                    signals.append(f"营收同比+{rev_yoy:.1f}%，稳健增长")
                elif rev_yoy < -10:
                    score -= 12
                    signals.append(f"营收同比{rev_yoy:.1f}%，业绩下滑")

            if profit_yoy is not None:
                if profit_yoy > 30:
                    score += 10
                elif profit_yoy > 15:
                    score += 5
                elif profit_yoy < -20:
                    score -= 15
                    signals.append(f"净利润同比{profit_yoy:.1f}%，大幅下滑")

            # 偿债能力
            debt_ratio = financials.get("debt_ratio")
            current_ratio = financials.get("current_ratio")

            details["debt_ratio"] = debt_ratio
            details["current_ratio"] = current_ratio

            if debt_ratio is not None:
                if debt_ratio < DEBT_RATIO_LOW:
                    score += 5
                    signals.append(f"资产负债率{debt_ratio:.1f}%，财务稳健")
                elif debt_ratio > DEBT_RATIO_HIGH:
                    score -= 8
                    signals.append(f"资产负债率{debt_ratio:.1f}%，杠杆偏高")

            if current_ratio is not None:
                if current_ratio < CURRENT_RATIO_LOW:
                    score -= 5
                    signals.append(f"流动比率{current_ratio:.2f}，短期偿债压力")

            # 现金流
            ocf_ps = financials.get("ocf_per_share")
            eps = financials.get("eps")

            details["ocf_per_share"] = ocf_ps
            details["eps"] = eps

            if ocf_ps is not None and eps is not None and eps > 0:
                if ocf_ps > eps:
                    score += 5
                    signals.append("每股经营现金流>EPS，利润含金量高")
                elif ocf_ps < 0:
                    score -= 5
                    signals.append("经营现金流为负，需关注")

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"基本面分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("基本面", score, details=details, signals=signals)


class FinancialQualityAnalyzer:
    """维度五：财报质量分析 - 收入确认、现金流匹配度、非经常损益、审计意见"""

    @staticmethod
    def analyze(financials: dict) -> DimensionScore:
        details = {}
        signals = []
        score = 60.0  # 默认中等偏上

        try:
            if not financials or "error" in financials:
                return DimensionScore("财报质量", 40, details={"error": "无财务数据"})

            # 利润现金流匹配度
            ocf_ps = financials.get("ocf_per_share")
            eps = financials.get("eps")
            revenue = financials.get("revenue")
            net_profit = financials.get("net_profit")

            details["ocf_vs_eps_match"] = None
            if ocf_ps is not None and eps is not None and eps != 0:
                match_ratio = ocf_ps / eps
                details["ocf_vs_eps_match"] = round(match_ratio, 2)
                if match_ratio >= 1.0:
                    score += 10
                    signals.append("经营现金流覆盖净利润，利润质量高")
                elif match_ratio >= 0.5:
                    pass  # 正常
                elif match_ratio > 0:
                    score -= 8
                    signals.append("经营现金流低于净利润，利润含金量不足")
                else:
                    score -= 15
                    signals.append("经营现金流为负，利润质量存疑")

            # 收入与净利润匹配
            if revenue and revenue > 0 and net_profit is not None:
                net_margin = net_profit / revenue * 100
                if net_margin < 1 and net_profit > 0:
                    score -= 5
                    signals.append(f"净利率仅{net_margin:.2f}%，盈利微薄")

            # ROE稳定性（通过ROA辅助判断）
            roa = financials.get("roa")
            roe = financials.get("roe")
            if roa is not None and roe is not None:
                leverage_effect = roe - roa
                if leverage_effect > 15:
                    score -= 5
                    signals.append("ROE主要依赖杠杆驱动而非经营效率")

            # EPS趋势
            if eps is not None:
                if eps <= 0:
                    score -= 15
                    signals.append("EPS为负或零")
                elif eps < 0.1:
                    score -= 5
                    signals.append("EPS偏低，盈利能力弱")

            # 总资产周转（间接判断运营效率）
            total_assets = financials.get("total_assets")
            revenue_val = financials.get("revenue")
            if total_assets and total_assets > 0 and revenue_val:
                asset_turnover = revenue_val / total_assets
                details["asset_turnover"] = round(asset_turnover, 2)
                if asset_turnover > 0.8:
                    score += 5
                elif asset_turnover < 0.3:
                    score -= 3

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"财报质量分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("财报质量", score, details=details, signals=signals)


class SentimentAnalyzer:
    """维度六：舆情情绪分析 - 新闻情绪、市场关注度、事件驱动"""

    @staticmethod
    def analyze(news: dict, quote: dict) -> DimensionScore:
        details = {}
        signals = []
        score = 50.0

        try:
            if news and "error" not in news:
                sentiment = news.get("sentiment", "neutral")
                news_count = news.get("news_count", 0)
                pos_score = news.get("positive_score", 0)
                neg_score = news.get("negative_score", 0)

                details["sentiment"] = sentiment
                details["news_count_7d"] = news_count
                details["positive_keywords"] = pos_score
                details["negative_keywords"] = neg_score

                # 情绪方向
                if sentiment == "positive":
                    score += 15
                    signals.append(f"近7日{news_count}条新闻，情绪偏正面")
                elif sentiment == "negative":
                    score -= 15
                    signals.append(f"近7日{news_count}条新闻，情绪偏负面")
                elif sentiment == "mixed":
                    score -= 3
                    signals.append("新闻情绪多空交织，存在分歧")

                # 关注度（新闻数量本身反映热度）
                if news_count > 30:
                    score += 3
                    signals.append(f"{news_count}条新闻，市场高度关注")
                elif news_count == 0:
                    details["note"] = "近7日无新闻，关注度低"

                # 关键词强度
                total_sent = pos_score + neg_score
                if total_sent > 10:
                    intensity = abs(pos_score - neg_score) / total_sent
                    details["sentiment_intensity"] = round(intensity, 2)
                    if intensity > 0.7:
                        signals.append("情绪信号较强，值得关注")

                # 提取关键标题
                headlines = news.get("headlines", [])
                if headlines:
                    details["top_headlines"] = [h["title"] for h in headlines[:5]]
            else:
                details["note"] = "无舆情数据"

            # 换手率作为市场热度补充
            turnover = quote.get("turnover_rate", 0)
            if turnover:
                details["turnover_rate"] = turnover
                if turnover > TURN_OVER_HEAVY:
                    score += 3
                    signals.append(f"换手率{turnover:.1f}%，交易活跃")
                elif turnover > TURN_OVER_SPECULATIVE:
                    signals.append(f"换手率{turnover:.1f}%，投机氛围浓，注意风险")

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"舆情分析异常: {e}")
            details["error"] = str(e)

        score = min(max(score, 0), 100)
        return DimensionScore("舆情情绪", score, details=details, signals=signals)


# ==================== 综合分析入口 ====================

class StockAnalyzer:
    """综合分析器 - 汇聚六个维度，输出完整报告"""

    def __init__(self, provider: AStockDataProvider = None):
        self.provider = provider or AStockDataProvider()

    def analyze(self, symbol: str) -> dict:
        """对单只股票进行完整的六维分析

        Args:
            symbol: 股票代码

        Returns:
            完整分析报告(dict)
        """
        report = {
            "symbol": symbol,
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_score": 0,
            "overall_rating": "",
            "action_suggestion": "",
            "dimensions": [],
            "key_signals": [],
            "risk_warnings": [],
        }

        # 并行拉取所有数据
        quote = self.provider.get_realtime_quote(symbol)
        kline = self.provider.get_history_kline(symbol, period="daily")

        # ETF 快速通道: 跳过财务/基本面/资金流, 仅做技术面+通达信
        # 上交所: 5xxxxx(ETF/LOF), 159xxx(深市ETF)
        s6 = str(symbol).zfill(6)
        is_etf = s6.startswith("5") or s6.startswith("15")

        if is_etf:
            report["overall_score"] = 0
            report["overall_rating"] = "N/A"
            report["action_suggestion"] = "ETF品种,仅显示技术面"
            report["dimensions"] = []
            report["key_signals"] = []
            report["risk_warnings"] = []
            report["quote"] = quote
            # 仅做通达信指标分析
            try:
                try:
                    from .tdx_indicators import tdx_combined_analysis
                except ImportError:
                    from tdx_indicators import tdx_combined_analysis
                report["tdx_indicators"] = tdx_combined_analysis(kline)
            except Exception as e:
                logger.warning(f"ETF通达信分析失败 {symbol}: {e}")
                report["tdx_indicators"] = {"error": str(e)}
            try:
                try:
                    from .tdx_signal_catcher import tdx_signal_catcher
                except ImportError:
                    from tdx_signal_catcher import tdx_signal_catcher
                report["tdx_signals"] = tdx_signal_catcher(kline).to_dict()
            except Exception as e:
                logger.warning(f"ETF信号分析失败 {symbol}: {e}")
                report["tdx_signals"] = {"error": str(e)}
            return report

        money_flow = self.provider.get_money_flow(symbol)
        lhb = self.provider.get_lhb(symbol)
        financials = self.provider.get_financials(symbol)
        valuation = self.provider.get_valuation_percentile(symbol)
        news = self.provider.get_news_sentiment(symbol)

        # 六维分析
        tech = TechnicalAnalyzer.analyze(kline)
        val = ValuationAnalyzer.analyze(quote, valuation, financials)
        mf = MoneyFlowAnalyzer.analyze(money_flow, lhb, quote)
        fund = FundamentalAnalyzer.analyze(financials, quote)
        fq = FinancialQualityAnalyzer.analyze(financials)
        sent = SentimentAnalyzer.analyze(news, quote)

        dimensions = [tech, val, mf, fund, fq, sent]
        report["dimensions"] = [d.to_dict() for d in dimensions]

        # 加权综合得分
        weights = {"技术面": 0.20, "估值": 0.18, "资金面": 0.22,
                   "基本面": 0.18, "财报质量": 0.10, "舆情情绪": 0.12}
        overall = sum(d.score * weights.get(d.name, 1/6) for d in dimensions)
        report["overall_score"] = round(overall, 1)
        report["overall_rating"] = _score_to_rating(overall)

        # 收集关键信号
        all_signals = []
        for d in dimensions:
            all_signals.extend(d.signals)
        report["key_signals"] = all_signals[:15]

        # 风险提示
        warnings = []
        for d in dimensions:
            if d.score < 30:
                warnings.append(f"{d.name}评分仅{d.score:.0f}分，需重点关注")
            if "error" in d.details:
                warnings.append(f"{d.name}数据缺失: {d.details['error']}")
        report["risk_warnings"] = warnings

        # === 扩展分析: 多策略共振 ===
        try:
            try:
                from .resonance import ResonanceEngine
            except ImportError:
                from resonance import ResonanceEngine
            resonance = ResonanceEngine().analyze(kline, symbol)
            report["resonance"] = {
                "score": resonance.resonance_score,
                "direction": resonance.dominant_direction,
                "confidence": resonance.confidence,
                "convergent": resonance.convergent_signals,
                "recommendation": resonance.recommendation,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["resonance"] = None

        # === 扩展分析: 多空辩论 ===
        try:
            try:
                from .debate import DebateEngine
            except ImportError:
                from debate import DebateEngine
            debate = DebateEngine().debate(quote, kline, financials, valuation, money_flow, news, lhb)
            report["debate"] = {
                "bull_score": debate.bull_score,
                "bear_score": debate.bear_score,
                "verdict": debate.verdict,
                "confidence": debate.confidence,
                "summary": debate.summary,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["debate"] = None

        # === 扩展分析: 缠论笔段 ===
        try:
            try:
                from .chan import ChanAnalyzer
            except ImportError:
                from chan import ChanAnalyzer
            chan = ChanAnalyzer().analyze(kline)
            report["chan"] = {
                "position": chan.current_position,
                "buy_point": chan.buy_point,
                "sell_point": chan.sell_point,
                "buy_price": chan.buy_price,
                "sell_price": chan.sell_price,
                "structure_score": chan.structure_score,
                "recommendation": chan.recommendation,
                "warnings": chan.warnings,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["chan"] = None

        # === 扩展分析: 价值投资深度 ===
        try:
            try:
                from .value_investing import ValueInvestingAnalyzer
            except ImportError:
                from value_investing import ValueInvestingAnalyzer
            vi = ValueInvestingAnalyzer().analyze(quote, financials, valuation, symbol)
            report["value_investing"] = {
                "intrinsic_value": vi.intrinsic_value,
                "margin_of_safety": vi.margin_of_safety,
                "value_verdict": vi.value_verdict,
                "moat_score": vi.moat_score,
                "moat_level": vi.moat_level,
                "investment_case": vi.investment_case,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["value_investing"] = None

        # === 扩展分析: 事件风险 ===
        try:
            try:
                from .event_risk import EventRiskAnalyzer
            except ImportError:
                from event_risk import EventRiskAnalyzer
            er = EventRiskAnalyzer().analyze(quote, financials, news, symbol)
            report["event_risk"] = {
                "overall_risk": er.overall_risk,
                "risk_score": er.risk_score,
                "summary": er.summary,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["event_risk"] = None

        # === 扩展分析: 通达信指标(暗盘金/机构活跃度/BBI/LON/操盘量能) ===
        try:
            try:
                from .tdx_indicators import tdx_combined_analysis
            except ImportError:
                from tdx_indicators import tdx_combined_analysis
            tdx = tdx_combined_analysis(kline)
            report["tdx_indicators"] = {
                "combined_verdict": tdx.combined_verdict,
                "confidence": tdx.confidence,
                "recommendation": tdx.recommendation,
                "signals": tdx.overall_signals,
                "dark_money": {
                    "intensity": tdx.dark_money.dark_intensity,
                    "dm_3d": tdx.dark_money.dark_money_3d,
                    "dm_5d": tdx.dark_money.dark_money_5d,
                    "trend": tdx.dark_money.trend,
                },
                "inst_activity": {
                    "level": tdx.inst_activity.level,
                    "score": tdx.inst_activity.activity_score,
                    "signals": tdx.inst_activity.key_signals,
                },
                "bbi": {
                    "bbi_value": tdx.bbi_signals.bbi,
                    "position": tdx.bbi_signals.bb_position,
                    "gs_signal": tdx.bbi_signals.gs_signal,
                    "ema_alignment": tdx.bbi_signals.trend_alignment,
                },
                "lon": {
                    "signal": tdx.lon_trend.signal,
                    "value": tdx.lon_trend.lon_value,
                    "trend": tdx.lon_trend.lon_trend,
                },
                "volume_ops": {
                    "pattern": tdx.volume_ops.volume_pattern,
                    "buy_ratio": tdx.volume_ops.buy_ratio,
                    "condition": tdx.volume_ops.condition,
                    "macd": tdx.volume_ops.macd_signal,
                },
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            report["tdx_indicators"] = None

        # === 扩展分析: 提前抄底信号 ===
        try:
            try:
                from .tdx_signal_catcher import tdx_signal_catcher
            except ImportError:
                from tdx_signal_catcher import tdx_signal_catcher
            sc = tdx_signal_catcher(kline)
            report["tdx_signals"] = sc.to_dict()
        except (ImportError, AttributeError, TypeError, ValueError):
            report["tdx_signals"] = None

        # 操作建议
        report["action_suggestion"] = self._generate_action(
            overall, dimensions, quote
        )

        # 附带原始数据摘要
        report["data_snapshot"] = {
            "quote": {k: v for k, v in quote.items() if k != "timestamp"} if isinstance(quote, dict) else quote,
            "money_flow_main_net": money_flow.get("main_net_inflow") if isinstance(money_flow, dict) else None,
            "sentiment": news.get("sentiment") if isinstance(news, dict) else None,
        }

        return report

    def batch_analyze(self, symbols: list) -> list:
        """批量分析多只股票"""
        results = []
        for sym in symbols:
            try:
                r = self.analyze(sym)
                results.append(r)
            except (ValueError, TypeError, KeyError) as e:
                results.append({"symbol": sym, "error": str(e)})
        return results

    def _generate_action(self, overall_score: float, dimensions: list, quote: dict) -> str:
        """根据综合得分生成操作建议"""
        actions = []

        if overall_score >= 75:
            actions.append("【建议】综合评估优秀，可考虑逢低建仓或加仓")
        elif overall_score >= 60:
            actions.append("【建议】综合评估良好，可持有或轻仓参与")
        elif overall_score >= 45:
            actions.append("【建议】综合评估中性，建议观望或小仓位试探")
        elif overall_score >= 30:
            actions.append("【建议】综合评估偏空，建议减仓或回避")
        else:
            actions.append("【建议】综合评估较差，建议回避或等待企稳信号")

        # 补充具体维度建议
        tech_d = next((d for d in dimensions if d.name == "技术面"), None)
        mf_d = next((d for d in dimensions if d.name == "资金面"), None)
        val_d = next((d for d in dimensions if d.name == "估值"), None)

        if tech_d and tech_d.score >= 75:
            actions.append("技术形态向好，短线可积极跟进")
        elif tech_d and tech_d.score <= 30:
            actions.append("技术面走弱，注意止损")

        if mf_d and mf_d.score >= 70:
            actions.append("资金面支撑强，中期有上行动力")
        elif mf_d and mf_d.score <= 30:
            actions.append("资金持续流出，不宜逆势操作")

        if val_d and val_d.score >= 70:
            actions.append("估值具备安全边际，适合中长线布局")
        elif val_d and val_d.score <= 30:
            actions.append("估值偏高，追高风险大于收益")

        price = quote.get("price", 0) if isinstance(quote, dict) else 0
        if price > 0:
            # 简单止盈止损参考
            actions.append(f"当前价: {price:.2f} | 参考止损位: {price * 0.92:.2f}(-8%) | 参考止盈位: {price * 1.15:.2f}(+15%)")

        return "\n".join(actions)
