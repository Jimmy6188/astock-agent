"""
通达信指标翻译模块 (TDX Indicators Port)
将5个通达信主图/副图指标翻译为Python，复用已有OHLCV数据。

支持指标:
  1. dark_money()    - 暗盘金副图 (主力暗盘资金流)
  2. inst_activity() - 机构活跃度副图
  3. bbi_signals()   - BBI主图关键信号 (GS策略/EMA排列/试盘起爆)
  4. lon_trend()     - LON副图 (资金趋势+差分振荡器)
  5. volume_ops()    - 操盘量能副图 (买盘卖盘分解+量价情境)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


# ============================================================
# 1. 暗盘金副图 (Dark Money Flow)
# ============================================================

@dataclass
class DarkMoneyResult:
    """暗盘资金流分析结果"""
    dark_money_3d: float      # 3日累计暗盘资金(万)
    dark_money_5d: float      # 5日累计暗盘资金(万)
    dark_intensity: str       # "强流入"/"流入"/"中性"/"流出"/"强流出"
    x8_score: float           # X_8强度分(当日)
    trend: str               # "持续流入"/"持续流出"/"交替"


def dark_money(kline: pd.DataFrame) -> DarkMoneyResult:
    """
    暗盘资金流分析 (通达信暗盘金副图)

    原理: 通过开盘/收盘/最高/最低相对前收的6个比率求和，
    构建"X_7复合比率"，再乘以当日成交额估算主力暗盘资金流。

    通达信原公式核心:
      X_1 = (O-REF(C,1))/REF(C,1)
      X_2 = (C-O)/O
      X_3 = (H-O)/O
      X_4 = (C-H)/H
      X_5 = (L-O)/O
      X_6 = (C-L)/L
      X_7 = X_1+X_2+X_3+X_4+X_5+X_6
      X_8 = IF(X_7>=1, 0.8, X_7)
      暗盘资金 = AMOUNT * X_8 / 1e8
    """
    if kline is None or kline.empty or len(kline) < 5:
        return DarkMoneyResult(0, 0, "中性", 0, "数据不足")

    df = kline.tail(5).copy().reset_index(drop=True)
    prev_close = kline['close'].values[-6] if len(kline) > 5 else df['close'].values[0]

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    amounts = df['amount'].values if 'amount' in df.columns else None

    # 计算各日X_8
    x8_values = []
    dm_values = []

    for i in range(len(df)):
        pc = kline['close'].values[-6 + i] if len(kline) > 5 and i < 5 else prev_close
        if i >= 1 and len(kline) > 5:
            pc = kline['close'].values[-6 + i]
        elif i >= 1:
            pc = closes[i-1]
        else:
            pc = closes[0]

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        if pc == 0 or o == 0 or h == 0 or l == 0:
            x8_values.append(0)
            dm_values.append(0)
            continue

        x1 = (o - pc) / pc
        x2 = (c - o) / o
        x3 = (h - o) / o
        x4 = (c - h) / h
        x5 = (l - o) / o
        x6 = (c - l) / l

        x7 = x1 + x2 + x3 + x4 + x5 + x6
        x8 = min(x7, 0.8) if x7 >= 1 else x7

        # 暗盘资金(万元)
        if amounts is not None and amounts[i] > 0:
            dm = amounts[i] * abs(x8) / 1e8 if x8 > 0 else (-amounts[i] * abs(x8) / 1e8)
        else:
            # 无成交额时用VOL*CLOSE估算
            vol = df['volume'].values[i] if 'volume' in df.columns else 0
            dm = vol * c * x8 / 1e8

        x8_values.append(x8)
        dm_values.append(dm)

    x8_current = x8_values[-1] if x8_values else 0
    dm_3d = sum(dm_values[-3:]) if len(dm_values) >= 3 else sum(dm_values)
    dm_5d = sum(dm_values)

    # 强度判定
    if x8_current > 0.5:
        intensity = "强流入"
    elif x8_current > 0.1:
        intensity = "流入"
    elif x8_current > -0.1:
        intensity = "中性"
    elif x8_current > -0.5:
        intensity = "流出"
    else:
        intensity = "强流出"

    # 趋势
    if len(dm_values) >= 3:
        if all(v > 0 for v in dm_values[-3:]):
            trend = "持续流入"
        elif all(v < 0 for v in dm_values[-3:]):
            trend = "持续流出"
        else:
            trend = "交替"
    else:
        trend = "数据不足"

    return DarkMoneyResult(
        dark_money_3d=round(dm_3d, 2),
        dark_money_5d=round(dm_5d, 2),
        dark_intensity=intensity,
        x8_score=round(x8_current, 4),
        trend=trend,
    )


# ============================================================
# 2. 机构活跃度副图 (Institutional Activity)
# ============================================================

@dataclass
class InstActivityResult:
    """机构活跃度分析结果"""
    activity_score: float       # 活跃度分数
    level: str                  # "大牛"/"强势"/"生命线以上"/"弱势"
    is_institutional: bool      # 是否达到机构活跃度阈值
    kline_strength: float       # K线形态综合强度
    key_signals: List[str]      # 关键信号


def inst_activity(kline: pd.DataFrame) -> InstActivityResult:
    """
    机构活跃度分析 (通达信机构活跃度副图)

    通达信原公式核心:
      X_5 = (IF(C<=O,C,O)-L)/L*100        # 下影线幅度
      X_6 = (C-REF(C,1))/REF(C,1)*100      # 涨跌幅
      X_7 = (O-REF(C,1))/REF(C,1)*100      # 开盘跳空幅度
      X_8 = (C-O)/O*100                    # 实体幅度
      X_10 = (H-IF(C>=O,C,O))/IF(C>=O,C,O)*100  # 上影线幅度

    取这6个维度的最大值*1.2作为活跃度
    阈值: 1.56(生命线) / 3(强势线) / 6(大牛线)
    """
    if kline is None or kline.empty or len(kline) < 2:
        return InstActivityResult(0, "弱势", False, 0, ["数据不足"])

    df = kline.tail(5).copy().reset_index(drop=True)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    prev_closes = kline['close'].values[-5:]

    signals = []
    max_strength = 0
    max_level = "弱势"

    for i in range(len(df)):
        c, o, h, l = closes[i], opens[i], highs[i], lows[i]
        pc = prev_closes[i] if i < len(prev_closes) else closes[0]

        if pc == 0 or l == 0 or o == 0:
            continue

        # 下影线幅度
        lower_shadow = ((min(c, o) - l) / l) * 100

        # 涨跌幅
        change_pct = ((c - pc) / pc) * 100

        # 开盘跳空幅度
        gap_pct = ((o - pc) / pc) * 100

        # 实体幅度
        body_pct = ((c - o) / o) * 100

        # 上影线幅度
        upper_shadow = ((h - max(c, o)) / max(c, o)) * 100

        # 振幅
        amplitude = ((h - l) / l) * 100

        # 取最大值*1.2
        max_val = max(abs(lower_shadow), abs(change_pct), abs(gap_pct),
                       abs(body_pct), abs(upper_shadow), amplitude)
        activity = max_val * 1.2

        max_strength = max(max_strength, activity)

        # 判定级别
        if activity >= 6:
            level = "大牛"
            signals.append(f"大牛线级别活跃度({activity:.1f})")
        elif activity >= 3:
            level = "强势"
            if not signals or "强势" not in signals[-1]:
                signals.append(f"强势线级别活跃度({activity:.1f})")
        elif activity >= 1.56:
            level = "生命线以上"
        else:
            level = "弱势"

    # 机构活跃度判定(接近涨停且活跃度>5)
    is_inst = max_strength > 5

    if max_strength >= 6:
        final_level = "大牛"
    elif max_strength >= 3:
        final_level = "强势"
    elif max_strength >= 1.56:
        final_level = "生命线以上"
    else:
        final_level = "弱势"

    return InstActivityResult(
        activity_score=round(max_strength, 2),
        level=final_level,
        is_institutional=is_inst,
        kline_strength=round(max_strength, 2),
        key_signals=signals[:5],
    )


# ============================================================
# 3. BBI主图关键信号
# ============================================================

@dataclass
class BBISignalResult:
    """BBI主图关键信号"""
    bbi: float                # BBI均线值
    bb_position: str          # "站上BBI"/"跌破BBI"/"缠绕"
    gs_signal: str            # "GS买"/"GS卖"/"无信号"
    trend_alignment: str      # "多头排列"/"空头排列"/"混乱"
    ema_score: float          # EMA趋势得分(0~100)
    trial_signal: bool        # 是否有试盘信号
    explosion_signal: bool    # 是否有起爆信号
    key_signals: List[str]


def bbi_signals(kline: pd.DataFrame) -> BBISignalResult:
    """
    BBI主图关键信号提取

    通达信原公式核心:
      BBI = (MA(C,3)+MA(C,7)+MA(C,13)+MA(C,27))/4
      GS策略: A0=(H+L+2O+6C)/10, BB=BBI
        TK = 空头K线形态
        TP = 多头K线形态
        C1 = CROSS(A0,BB) AND TK  → 卖信号
        C2 = CROSS(BB,A0) AND TP  → 买信号
        迭代8次过滤假信号

      EMA趋势排列: EMA5>EMA6>EMA7>...>EMA250
    """
    if kline is None or kline.empty or len(kline) < 27:
        return BBISignalResult(0, "缠绕", "无信号", "混乱", 0, False, False, ["数据不足"])

    df = kline.copy()
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(closes)

    signals = []

    # --- BBI均线 ---
    ma3 = np.convolve(closes, np.ones(3)/3, mode='valid')
    ma7 = np.convolve(closes, np.ones(7)/7, mode='valid')
    ma13 = np.convolve(closes, np.ones(13)/13, mode='valid')
    ma27 = np.convolve(closes, np.ones(27)/27, mode='valid')

    # 对齐长度
    len3, len7, len13, len27 = len(ma3), len(ma7), len(ma13), len(ma27)
    # 取最短长度
    min_len = min(len3, len7, len13, len27)
    if min_len < 2:
        return BBISignalResult(0, "缠绕", "无信号", "混乱", 0, False, False, ["数据不足"])

    bbi = (ma3[-min_len:] + ma7[-min_len:] + ma13[-min_len:] + ma27[-min_len:]) / 4
    cur_bbi = bbi[-1]
    cur_close = closes[-1]

    if cur_close > cur_bbi * 1.01:
        bb_pos = "站上BBI"
    elif cur_close < cur_bbi * 0.99:
        bb_pos = "跌破BBI"
    else:
        bb_pos = "缠绕"

    # --- GS策略 (简化版: 单次CROSS检测) ---
    a0 = (highs + lows + 2*opens + 6*closes) / 10

    # TK: 空头K线
    tk = (
        (closes < opens) |
        ((closes < np.roll(highs, -1)) & (closes > opens)) |
        ((closes >= opens) & ((highs - closes) >= (closes - opens)) & (closes / np.maximum(np.roll(closes, 1), 0.01) < 1.02)) |
        ((closes == opens) & ((highs - closes) >= (closes - lows)) & (closes / np.maximum(np.roll(closes, 1), 0.01) < 1.05))
    )

    # TP: 多头K线
    tp = (
        ((closes > opens) & (closes / np.maximum(np.roll(closes, 1), 0.01) > 0.94)) |
        ((closes > np.roll(lows, 1)) & (closes < opens)) |
        ((closes <= opens) & ((closes - lows) >= (opens - closes)) & (closes / np.maximum(np.roll(closes, 1), 0.01) > 0.98)) |
        ((closes == opens) & ((closes - lows) >= (highs - closes)) & (closes / np.maximum(np.roll(closes, 1), 0.01) > 0.95))
    )

    # 检测CROSS信号(只看最近3天)
    gs_signal = "无信号"
    bbi_arr = bbi[-min_len:] if len(bbi) >= min_len else bbi[-len(bbi):]
    for i in range(max(1, n-3), n):
        if min_len <= 0 or len(bbi_arr) < 2:
            break
        # 找到当前K线对应的BBI值(相对位置)
        rel_idx = min(i - (n - min_len), min_len - 1)
        if rel_idx < 0:
            rel_idx = 0
        cur_bbi_val = float(bbi_arr[rel_idx])
        prev_bbi_val = float(bbi_arr[max(0, rel_idx - 1)])

        if a0[i-1] < prev_bbi_val and a0[i] > cur_bbi_val and tk[i]:
            gs_signal = "GS卖"
            break
        if a0[i-1] > prev_bbi_val and a0[i] < cur_bbi_val and tp[i]:
            gs_signal = "GS买"
            break

    # --- EMA趋势排列 ---
    emas = {}
    for period in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 60, 90, 120]:
        if n >= period:
            emas[period] = pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

    sorted_emas = sorted(emas.items(), key=lambda x: x[0])
    aligned_up = all(sorted_emas[i][1] > sorted_emas[i+1][1] for i in range(len(sorted_emas)-1))
    aligned_down = all(sorted_emas[i][1] < sorted_emas[i+1][1] for i in range(len(sorted_emas)-1))

    if aligned_up:
        trend = "多头排列"
        ema_score = 95
    elif aligned_down:
        trend = "空头排列"
        ema_score = 5
    else:
        # 计算实际排列度
        up_count = sum(1 for i in range(len(sorted_emas)-1) if sorted_emas[i][1] > sorted_emas[i+1][1])
        ema_score = (up_count / (len(sorted_emas)-1)) * 100
        trend = "多头排列" if ema_score >= 70 else ("空头排列" if ema_score <= 30 else "混乱")

    # --- 试盘/起爆信号 ---
    trial = False
    explosion = False

    if n >= 10:
        # 倍量检测
        vols = df['volume'].values if 'volume' in df.columns else None
        if vols is not None:
            for i in range(5, n):
                if i >= 1 and vols[i-1] > 0 and vols[i] / vols[i-1] > 2 and closes[i] > opens[i]:
                    # 试盘: 倍量阳线 + 突破近期高位
                    if highs[i] >= np.max(highs[max(0,i-6):i]):
                        trial = True
                        break

    if trial:
        signals.append("试盘信号: 倍量突破")
        # 起爆: 试盘后缩量洗盘再放量
        # (简化检测)

    if gs_signal != "无信号":
        signals.append(f"GS策略: {gs_signal}")

    if trend == "多头排列":
        signals.append("EMA多头排列，趋势向上")
    elif trend == "空头排列":
        signals.append("EMA空头排列，趋势向下")

    if bb_pos == "站上BBI":
        signals.append("价格站上BBI均线")
    elif bb_pos == "跌破BBI":
        signals.append("价格跌破BBI均线")

    return BBISignalResult(
        bbi=round(cur_bbi, 2),
        bb_position=bb_pos,
        gs_signal=gs_signal,
        trend_alignment=trend,
        ema_score=round(ema_score, 1),
        trial_signal=trial,
        explosion_signal=explosion,
        key_signals=signals[:5],
    )


# ============================================================
# 4. LON副图 (Long-term Volume Flow)
# ============================================================

@dataclass
class LONResult:
    """LON资金趋势"""
    lon_value: float          # LON当前值
    lon_trend: str            # "上升"/"下降"/"走平"
    diff: float               # DIFF
    dea: float                # DEA
    macd_bar: float           # MACD柱
    signal: str               # "金叉"/"死叉"/"多头"/"空头"


def lon_trend(kline: pd.DataFrame) -> LONResult:
    """
    LON资金趋势 (通达信Lon01副图)

    通达信原公式:
      LC = REF(CLOSE,1)
      VID = SUM(VOL,2) / (HHV(HIGH,2)-LLV(LOW,2)) * 100
      RC = (CLOSE-LC) * VID
      LONG = SUM(RC,0)              # 累加资金流
      DIFF = SMA(LONG,10,1)
      DEA = SMA(LONG,20,1)
      LON = DIFF - DEA
    """
    if kline is None or kline.empty or len(kline) < 5:
        return LONResult(0, "走平", 0, 0, 0, "多头")

    df = kline.copy()
    closes = df['close'].values
    vols = df['volume'].values if 'volume' in df.columns else np.zeros(len(closes))
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)

    # 计算LONG (累加资金流)
    prev_closes = np.concatenate([[closes[0]], closes[:-1]])

    long_vals = np.zeros(n)
    for i in range(n):
        lc = prev_closes[i] if i == 0 else closes[i-1]
        if i >= 1:
            hh = max(highs[i-1], highs[i])
            ll = min(lows[i-1], lows[i])
            vid = (vols[i-1] + vols[i]) / max((hh - ll), 0.01) * 100
        else:
            vid = 0

        rc = (closes[i] - lc) * vid
        long_vals[i] = rc if i == 0 else long_vals[i-1] + rc

    # DIFF = SMA(LONG,10,1)
    diff = pd.Series(long_vals).ewm(alpha=2/11, adjust=False).mean().values
    dea = pd.Series(long_vals).ewm(alpha=2/21, adjust=False).mean().values
    macd = diff - dea

    cur_lon = macd[-1]
    prev_diff = diff[-2] if len(diff) >= 2 else diff[-1]
    prev_dea = dea[-2] if len(dea) >= 2 else dea[-1]

    # 信号
    if diff[-1] > dea[-1] and prev_diff <= prev_dea:
        signal = "金叉"
    elif diff[-1] < dea[-1] and prev_diff >= prev_dea:
        signal = "死叉"
    elif diff[-1] > dea[-1]:
        signal = "多头"
    else:
        signal = "空头"

    # 趋势
    if cur_lon > 0:
        lon_trend = "上升"
    elif cur_lon < 0:
        lon_trend = "下降"
    else:
        lon_trend = "走平"

    return LONResult(
        lon_value=round(cur_lon, 2),
        lon_trend=lon_trend,
        diff=round(diff[-1], 2),
        dea=round(dea[-1], 2),
        macd_bar=round(cur_lon, 2),
        signal=signal,
    )


# ============================================================
# 5. 操盘量能副图 (Volume Operations)
# ============================================================

@dataclass
class VolumeOpsResult:
    """操盘量能分析"""
    buy_volume: float         # 买盘量
    sell_volume: float        # 卖盘量
    buy_ratio: float          # 买盘占比
    buy_sell_diff: float      # 买卖差
    volume_pattern: str       # "绿灯"/"衰退"/"伪装"/"蛰伏"
    condition: str            # 八种情境之一
    macd_signal: str          # MACD信号
    kdj_signal: str           # KDJ信号
    key_signals: List[str]


def volume_ops(kline: pd.DataFrame) -> VolumeOpsResult:
    """
    操盘量能分析 (通达信操盘量能副图)

    通达信原公式核心:
      WJ = (H+L+C)/3 (加权均价)
      V1~V4: 根据OHLC关系划分4个价格区间的量
      V5 = VOL / (H-L)  (量密度)
      买盘 = V8+V9 (下方买入量)
      卖盘 = V6+V7 (上方卖出量)
      MFI = (H-L)*1e6/(VOL*比)
      8种量价情境基于: 均线系统强弱 + 量比 + 相对大盘强弱
    """
    if kline is None or kline.empty or len(kline) < 5:
        return VolumeOpsResult(0, 0, 0, 0, "数据不足", "未知", "中性", "中性", ["数据不足"])

    df = kline.tail(20).copy().reset_index(drop=True)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = df['volume'].values if 'volume' in df.columns else np.zeros(len(closes))
    n = len(closes)
    signals = []

    # --- 买盘/卖盘分解 (最后一根K线) ---
    c, o, h, l = closes[-1], opens[-1], highs[-1], lows[-1]
    v = vols[-1]
    wj = (h + l + c) / 3

    if h == l:
        buy_vol = v * 0.5
        sell_vol = v * 0.5
    else:
        v_density = v / (h - l)

        v1 = max(0, h - max(o, c)) * v_density     # 上影线卖出
        v2 = max(0, max(c, o) - wj) * v_density     # 实体上半卖出
        v3 = max(0, wj - min(c, o)) * v_density     # 实体下半买入
        v4 = max(0, min(c, o) - l) * v_density      # 下影线买入

        sell_vol = v1 + v2
        buy_vol = v3 + v4

    buy_ratio = buy_vol / max(buy_vol + sell_vol, 0.01)
    bs_diff = buy_vol - sell_vol

    # --- MFI量价判断 ---
    prev_v = vols[-2] if n >= 2 and vols[-2] > 0 else 0
    prev_hl = max(highs[-2] - lows[-2], 0.01) if n >= 2 else 0.01
    cur_hl = max(h - l, 0.01)

    mfi_cur = cur_hl * 1e6 / max(v * 10, 0.01)     # 简化: 比=10
    mfi_prev = prev_hl * 1e6 / max(prev_v * 10, 0.01) if prev_v > 0 else 0

    if mfi_cur >= mfi_prev and v >= max(prev_v, 0.01):
        vol_pattern = "绿灯(量价齐升)"
    elif mfi_cur < mfi_prev and v < prev_v:
        vol_pattern = "衰退(量价齐跌)"
    elif mfi_cur >= mfi_prev and v < prev_v:
        vol_pattern = "伪装(价升量缩)"
    else:
        vol_pattern = "蛰伏(价跌量升)"

    # --- 均线强弱 ---
    if n >= 20:
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        prev_ma20 = np.mean(closes[-21:-1])

        yyavx = 0
        yyavx += 10 if c > ma5 else -10
        yyavx += 10 if ma5 > ma10 else -10
        yyavx += 10 if c > ma10 else -10
        yyavx += 10 if ma5 > ma20 else -10
        yyavx += 10 if c > ma20 else -10
        yyavx += 10 if ma20 > prev_ma20 else -10

        vol5 = np.mean(vols[-5:])
        vol10 = np.mean(vols[-10:])

        # 判断八种情境
        if yyavx > 0 and vol5 >= vol10:
            condition = "量能理想，明显走强，中线参与"
        elif yyavx > 0 and vol5 < vol10:
            condition = "走势趋强，量能不足，短线进场"
        elif yyavx > 0 and vol5 < vol10:
            condition = "量价良好，短线可进场"
        else:
            condition = "趋势不明，暂不参与"
    else:
        condition = "数据不足"
        yyavx = 0

    # --- MACD (12,26,9) ---
    if n >= 26:
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_val = (dif - dea) * 2

        if dif[-1] > dea[-1]:
            macd_sig = "多头"
        elif dif[-1] < dea[-1]:
            macd_sig = "空头"
        else:
            macd_sig = "中性"

        # KDJ (9,3,3)
        low9 = np.min(closes[-9:]) if n >= 9 else closes[-1]
        high9 = np.max(closes[-9:]) if n >= 9 else closes[-1]
        rsv = (closes[-1] - low9) / max(high9 - low9, 0.01) * 100
        k = np.mean([rsv, 50, 50])  # 简化
        if k > 80:
            kdj_sig = "超买"
        elif k < 20:
            kdj_sig = "超卖"
        else:
            kdj_sig = "中性"
    else:
        macd_sig = "中性"
        kdj_sig = "中性"

    if buy_ratio > 0.7:
        signals.append("买盘占优")
    elif buy_ratio < 0.3:
        signals.append("卖盘占优")

    signals.append(f"量价: {vol_pattern}")
    signals.append(f"MACD: {macd_sig}")
    signals.append(f"KDJ: {kdj_sig}")

    return VolumeOpsResult(
        buy_volume=round(buy_vol, 2),
        sell_volume=round(sell_vol, 2),
        buy_ratio=round(buy_ratio, 4),
        buy_sell_diff=round(bs_diff, 2),
        volume_pattern=vol_pattern,
        condition=condition,
        macd_signal=macd_sig,
        kdj_signal=kdj_sig,
        key_signals=signals,
    )


# ============================================================
# 综合分析器 (TDX综合分析)
# ============================================================

@dataclass
class TDXCombinedResult:
    """通达信指标综合分析结果"""
    dark_money: DarkMoneyResult
    inst_activity: InstActivityResult
    bbi_signals: BBISignalResult
    lon_trend: LONResult
    volume_ops: VolumeOpsResult
    combined_verdict: str       # "强烈看多"/"看多"/"中性"/"看空"/"强烈看空"
    confidence: str             # "高"/"中"/"低"
    overall_signals: List[str]
    recommendation: str


def tdx_combined_analysis(kline: pd.DataFrame) -> TDXCombinedResult:
    """
    通达信指标综合分析

    将5个指标的结论综合，给出统一的看多/看空判断。
    """
    dm = dark_money(kline)
    ia = inst_activity(kline)
    bbi = bbi_signals(kline)
    lon = lon_trend(kline)
    vo = volume_ops(kline)

    # 综合评分 (每个指标贡献20分)
    score = 50  # 基准

    # 暗盘资金
    if dm.dark_intensity in ("强流入", "流入"):
        score += 20 if "强" in dm.dark_intensity else 10
    elif dm.dark_intensity in ("流出", "强流出"):
        score -= 20 if "强" in dm.dark_intensity else 10

    # 机构活跃度
    if ia.level == "大牛":
        score += 20
    elif ia.level == "强势":
        score += 10
    elif ia.level == "弱势":
        score -= 5

    # BBI信号
    if bbi.bb_position == "站上BBI":
        score += 10
    elif bbi.bb_position == "跌破BBI":
        score -= 10
    if bbi.gs_signal == "GS买":
        score += 15
    elif bbi.gs_signal == "GS卖":
        score -= 15
    if bbi.trend_alignment == "多头排列":
        score += 10
    elif bbi.trend_alignment == "空头排列":
        score -= 10

    # LON趋势
    if lon.signal in ("金叉", "多头"):
        score += 10 if lon.signal == "多头" else 15
    elif lon.signal in ("死叉", "空头"):
        score -= 10 if lon.signal == "空头" else 15

    # 操盘量能
    if vo.volume_pattern.startswith("绿灯"):
        score += 10
    elif vo.volume_pattern.startswith("衰退"):
        score -= 10
    if vo.buy_ratio > 0.65:
        score += 5
    elif vo.buy_ratio < 0.35:
        score -= 5

    # 最终判定
    score = max(0, min(100, score))
    if score >= 75:
        verdict = "强烈看多"
        conf = "高"
    elif score >= 60:
        verdict = "看多"
        conf = "中"
    elif score >= 40:
        verdict = "中性"
        conf = "中"
    elif score >= 25:
        verdict = "看空"
        conf = "中"
    else:
        verdict = "强烈看空"
        conf = "高"

    # 关键信号汇总
    all_signals = []
    all_signals.append(f"暗盘资金:{dm.dark_intensity}({dm.dark_money_3d:+.1f}万)")
    all_signals.append(f"机构活跃度:{ia.level}({ia.activity_score})")
    all_signals.append(f"BBI:{bbi.bb_position}({bbi.gs_signal})")
    all_signals.append(f"EMA排列:{bbi.trend_alignment}")
    all_signals.append(f"LON:{lon.signal}({lon.macd_bar:+.1f})")
    all_signals.append(f"量价:{vo.volume_pattern}")
    all_signals.append(f"买卖比:{vo.buy_ratio*100:.0f}%/{100-vo.buy_ratio*100:.0f}%")

    recommendation = f"通达信指标综合判定: {verdict}(置信度:{conf}) | 综合得分:{score}"

    return TDXCombinedResult(
        dark_money=dm,
        inst_activity=ia,
        bbi_signals=bbi,
        lon_trend=lon,
        volume_ops=vo,
        combined_verdict=verdict,
        confidence=conf,
        overall_signals=all_signals,
        recommendation=recommendation,
    )