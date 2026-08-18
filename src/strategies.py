"""
内置交易策略定义
支持: 双均线 / MACD金叉死叉 / 布林带突破 / 网格交易 / 动量突破 / 海龟交易法
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Signal:
    """交易信号"""
    action: str  # "BUY" / "SELL" / "HOLD"
    strength: float = 0.0  # 信号强度 0~1
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """策略参数配置"""
    name: str = ""
    params: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig(name=self.__class__.__name__)
        self.name = self.config.name

    @abstractmethod
    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        """生成信号序列

        Args:
            kline_df: K线数据，需包含 date/open/high/low/close/volume 列

        Returns:
            原始K线附加 signal/action/strength/reason 列
        """
        pass

    @abstractmethod
    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        """获取最新信号

        Returns:
            当前交易信号
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """校验数据完整性"""
        required = {"date", "open", "high", "low", "close", "volume"}
        return required.issubset(df.columns) and len(df) >= 20


# ==================== 策略实现 ====================

class DualMAStrategy(BaseStrategy):
    """双均线交叉策略(MA5/MA20)

    默认参数:
    - fast_period: 5 (短期均线)
    - slow_period: 20 (长期均线)
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="双均线交叉",
            params={"fast_period": fast_period, "slow_period": slow_period}
        )
        super().__init__(config)
        self.fast = fast_period
        self.slow = slow_period

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(kline_df):
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)

        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()

        df["ma_fast"] = ma_fast
        df["ma_slow"] = ma_slow

        # 金叉/死叉判断
        df["diff"] = ma_fast - ma_slow
        df["signal_shift"] = df["diff"].shift(1)

        conditions = [
            (df["diff"] > 0) & (df["signal_shift"] <= 0),   # 金叉
            (df["diff"] < 0) & (df["signal_shift"] >= 0),   # 死叉
        ]
        choices = ["BUY", "SELL"]
        df["action"] = np.select(conditions, choices, default="HOLD")

        # 信号强度基于均线间距
        spread = abs(ma_fast - ma_slow) / ma_slow.replace(0, np.nan)
        df["strength"] = np.clip(spread.fillna(0) * 50, 0, 1)

        reasons = {
            "BUY": f"MA{self.fast}上穿MA{self.slow}金叉",
            "SELL": f"MA{self.fast}下穿MA{self.slow}死叉",
            "HOLD": "无信号",
        }
        df["reason"] = df["action"].map(reasons)

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()
        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "ma_fast": float(d.get("ma_fast", 0)) if pd.notna(d.get("ma_fast", np.nan)) else None,
                "ma_slow": float(d.get("ma_slow", 0)) if pd.notna(d.get("ma_slow", np.nan)) else None,
            }
        )


class MACDStrategy(BaseStrategy):
    """MACD经典策略

    参数:
    - fast: 12 (快线EMA)
    - slow: 26 (慢线EMA)
    - signal: 9 (信号线)
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="MACD",
            params={"fast": fast, "slow": slow, "signal": signal_period}
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(kline_df):
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)

        ema12 = close.ewm(span=self.fast).mean()
        ema26 = close.ewm(span=self.slow).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=self.signal_period).mean()
        macd_hist = 2 * (dif - dea)

        df["dif"] = dif
        df["dea"] = dea
        df["macd_hist"] = macd_hist

        # 金叉/死叉 + 柱状线方向
        df["dif_prev"] = dif.shift(1)
        df["dea_prev"] = dea.shift(1)

        conditions = [
            (dif > dea) & (df["dif_prev"] <= df["dea_prev"]),           # DIF上穿DEA
            (dif < dea) & (df["dif_prev"] >= df["dea_prev"]),           # DIF下穿DEA
            (dif > dea) & (macd_hist > macd_hist.shift(1)), # 多头强化
            (dif < dea) & (macd_hist < macd_hist.shift(1)), # 空头强化
        ]
        choices = ["BUY", "SELL", "BUY_STRONG", "SELL_STRONG"]
        action_map = {"BUY_STRONG": "BUY", "SELL_STRONG": "SELL"}
        df["action_raw"] = np.select(conditions, choices, default="HOLD")
        df["action"] = df["action_raw"].map(lambda x: action_map.get(x, x)).fillna("HOLD")

        # 强度
        strength_map = {"BUY": 0.6, "SELL": 0.6,
                        "BUY_STRONG": 1.0, "SELL_STRONG": 1.0, "HOLD": 0}
        df["strength"] = df["action_raw"].map(strength_map)

        reasons = {
            "BUY": "MACD金叉(DIF上穿DEA)",
            "SELL": "MACD死叉(DIF下穿DEA)",
            "BUY_STRONG": "MACD多头排列且柱状线增强",
            "SELL_STRONG": "MACD空头排列且柱状线增强",
            "HOLD": "MACD无明确信号",
        }
        df["reason"] = df["action_raw"].map(reasons)

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()
        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "dif": round(float(d.get("dif", 0)), 4) if pd.notna(d.get("dif", np.nan)) else None,
                "dea": round(float(d.get("dea", 0)), 4) if pd.notna(d.get("dea", np.nan)) else None,
                "macd_hist": round(float(d.get("macd_hist", 0)), 4) if pd.notna(d.get("macd_hist", np.nan)) else None,
            }
        )


class BollingerBreakoutStrategy(BaseStrategy):
    """布林带突破策略

    参数:
    - period: 20 (中轨周期)
    - std_mult: 2.0 (标准差倍数)
    """

    def __init__(self, period: int = 20, std_mult: float = 2.0,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="布林带突破",
            params={"period": period, "std_mult": std_mult}
        )
        super().__init__(config)
        self.period = period
        self.std_mult = std_mult

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(kline_df):
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)

        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.std_mult * std
        lower = mid - self.std_mult * std

        df["boll_mid"] = mid
        df["boll_upper"] = upper
        df["boll_lower"] = lower
        df["boll_width"] = (upper - lower) / mid.replace(0, np.nan)

        # 突破信号
        prev_close = close.shift(1)
        conditions = [
            (close > upper) & (prev_close <= upper.shift(1)),     # 上破
            (close < lower) & (prev_close >= lower.shift(1)),     # 下破
            (close < lower),                                      # 下轨下方
            (close > upper),                                      # 上轨上方
            (abs(close - mid) / std.replace(0, np.nan) < 0.5),   # 收口回归中轨
        ]
        choices = ["BUY_BREAKOUT", "SELL_BREAKOUT", "BUY_OVERSOLD",
                   "SELL_OVERBOUGHT", "HOLD_MEAN"]
        action_simplify = {
            "BUY_BREAKOUT": "BUY", "SELL_BREAKOUT": "SELL",
            "BUY_OVERSOLD": "BUY", "SELL_OVERBOUGHT": "SELL",
            "HOLD_MEAN": "HOLD"
        }
        df["action_raw"] = np.select(conditions, choices, default="HOLD")
        df["action"] = df["action_raw"].map(action_simplify).fillna("HOLD")

        strength_map = {
            "BUY_BREAKOUT": 0.9, "SELL_BREAKOUT": 0.9,
            "BUY_OVERSOLD": 0.7, "SELL_OVERBOUGHT": 0.7,
            "HOLD_MEAN": 0.3, "HOLD": 0
        }
        df["strength"] = df["action_raw"].map(strength_map)

        reasons = {
            "BUY_BREAKOUT": "价格突破布林上轨，强势上行",
            "SELL_BREAKOUT": "价格跌破布林下轨，强势下行",
            "BUY_OVERSOLD": "价格低于布林下轨，超卖反弹机会",
            "SELL_OVERBOUGHT": "价格高于布林上轨，超买回调风险",
            "HOLD_MEAN": "布林带收口，等待方向选择",
            "HOLD": "布林带内运行，无明显信号",
        }
        df["reason"] = df["action_raw"].map(reasons)

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()
        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "upper": round(float(d.get("boll_upper", 0)), 2),
                "mid": round(float(d.get("boll_mid", 0)), 2),
                "lower": round(float(d.get("boll_lower", 0)), 2),
                "width_pct": round(float(d.get("boll_width", 0)) * 100, 2) if pd.notna(d.get("boll_width", np.nan)) else None,
            }
        )


class GridTradingStrategy(BaseStrategy):
    """网格交易策略

    参数:
    - grid_size: 网格间距(%)
    - base_price: 基准价(默认为近期均价)
    - grid_count: 网格层数
    """

    def __init__(self, grid_size: float = 2.0, grid_count: int = 10,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="网格交易",
            params={"grid_size": grid_size, "grid_count": grid_count}
        )
        super().__init__(config)
        self.grid_size = grid_size / 100.0  # 转为小数
        self.grid_count = grid_count

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(kline_df):
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)

        # 以近60日均价为基准
        base_price = close.iloc[-60:].mean() if len(close) >= 60 else close.mean()

        df["base_price"] = base_price
        df["price_ratio"] = (close - base_price) / base_price

        # 计算当前所在网格层级
        df["grid_level"] = (df["price_ratio"] / self.grid_size).round(0).astype(int)

        # 触及网格线时产生信号
        df["grid_level_prev"] = df["grid_level"].shift(1)
        level_changed = df["grid_level"] != df["grid_level_prev"]

        conditions = [
            level_changed & (df["grid_level"] <= -2),      # 下跌触及买入网格
            level_changed & (df["grid_level"] >= 2),       # 上涨触及卖出网格
            (df["grid_level"] <= -3),                       # 深跌加仓区
            (df["grid_level"] >= 3),                        # 大涨减仓区
        ]
        choices = ["BUY_GRID", "SELL_GRID", "BUY_ADD", "SELL_REDUCE"]
        action_simple = {"BUY_GRID": "BUY", "SELL_GRID": "SELL",
                         "BUY_ADD": "BUY", "SELL_REDUCE": "SELL"}
        df["action_raw"] = np.select(conditions, choices, default="HOLD")
        df["action"] = df["action_raw"].map(action_simple).fillna("HOLD")
        abs_level = df["grid_level"].abs().astype(float)
        df["strength"] = np.clip(abs_level / self.grid_count, 0, 1)

        reasons = {
            "BUY_GRID": f"下跌{self.grid_size*100:.0f}%触及买入网格线",
            "SELL_GRID": f"上涨{self.grid_size*100:.0f}%触及卖出网格线",
            "BUY_ADD": "深跌区域，建议加倍买入",
            "SELL_REDUCE": "大涨区域，建议减仓止盈",
            "HOLD": "在网格区间内运行，持有",
        }
        df["reason"] = df["action_raw"].map(reasons)

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()
        current_level = int(d.get("grid_level", 0))
        base = float(d.get("base_price", 0))

        buy_grids = [round(base * (1 - i * self.grid_size), 2)
                     for i in range(1, self.grid_count + 1)]
        sell_grids = [round(base * (1 + i * self.grid_size), 2)
                      for i in range(1, self.grid_count + 1)]

        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "current_level": current_level,
                "base_price": round(base, 2),
                "buy_grid_levels": buy_grids[:5],
                "sell_grid_levels": sell_grids[:5],
                "grid_size_pct": f"{self.grid_size*100:.1f}%",
            }
        )


class MomentumBreakoutStrategy(BaseStrategy):
    """动量突破策略(结合成交量确认)

    参数:
    - lookback: 回看天数(计算突破基准)
    - volume_mult: 成交量放大倍数阈值
    """

    def __init__(self, lookback: int = 20, volume_mult: float = 1.5,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="动量突破",
            params={"lookback": lookback, "volume_mult": volume_mult}
        )
        super().__init__(config)
        self.lookback = lookback
        self.volume_mult = volume_mult

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(kline_df) or len(kline_df) < self.lookback + 5:
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # 近N日高低点
        rolling_high = high.rolling(self.lookback).max()
        rolling_low = low.rolling(self.lookback).min()
        vol_ma = volume.rolling(self.lookback).mean()

        df["resist_high"] = rolling_high
        df["support_low"] = rolling_low
        df["vol_ma"] = vol_ma
        df["vol_ratio"] = volume / vol_ma.replace(0, np.nan)

        # 突破条件
        conditions = [
            (close > rolling_high.shift(1)) &
            (volume > vol_ma * self.volume_mult),          # 放量突破前高
            (close < rolling_low.shift(1)) &
            (volume > vol_ma * self.volume_mult),          # 放量跌破前低
            (close > rolling_high.shift(1)) &
            (volume <= vol_ma * self.volume_mult),         # 缩量突破(假突破可能)
            (close < rolling_low.shift(1)) &
            (volume <= vol_ma * self.volume_mult),         # 缩量跌破
        ]
        choices = ["BUY_MOM", "SELL_MOM", "BUY_WEAK", "SELL_WEAK"]
        action_simple = {"BUY_MOM": "BUY", "SELL_MOM": "SELL",
                         "BUY_WEAK": "HOLD", "SELL_WEAK": "HOLD"}
        df["action_raw"] = np.select(conditions, choices, default="HOLD")
        df["action"] = df["action_raw"].map(action_simple).fillna("HOLD")

        strength_map = {"BUY_MOM": 0.85, "SELL_MOM": 0.85,
                        "BUY_WEAK": 0.3, "SELL_WEAK": 0.3, "HOLD": 0}
        df["strength"] = df["action_raw"].map(strength_map)

        reasons = {
            "BUY_MOM": f"放量突破近{self.lookback}日高点，动量强劲",
            "SELL_MOM": f"放量跌破近{self.lookback}日低点，动量向下",
            "BUY_WEAK": f"缩量突破近{self.lookback}日高点，需观察确认",
            "SELL_WEAK": f"缩量跌破近{self.lookback}日低点，需观察确认",
            "HOLD": "未出现有效突破信号",
        }
        df["reason"] = df["action_raw"].map(reasons)

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()
        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "resist_high": round(float(d.get("resist_high", 0)), 2),
                "support_low": round(float(d.get("support_low", 0)), 2),
                "vol_ratio": round(float(d.get("vol_ratio", 0)), 2) if pd.notna(d.get("vol_ratio", np.nan)) else None,
            }
        )


class TurtleStrategy(BaseStrategy):
    """海龟交易法简化版

    参数:
    - entry_window: 入市通道周期(默认20日)
    - exit_window: 离市通道周期(默认10日)
    - atr_mult: ATR倍数(止损用, 默认2.0)
    - max_risk_pct: 单笔最大风险敞口(默认1%)
    """

    def __init__(self, entry_window: int = 20, exit_window: int = 10,
                 atr_mult: float = 2.0, max_risk_pct: float = 0.01,
                 config: StrategyConfig = None):
        config = config or StrategyConfig(
            name="海龟交易法",
            params={
                "entry_window": entry_window,
                "exit_window": exit_window,
                "atr_mult": atr_mult,
                "max_risk_pct": max_risk_pct,
            }
        )
        super().__init__(config)
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_mult = atr_mult
        self.max_risk_pct = max_risk_pct

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR(真实波幅均值)"""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = df["close"].shift(1).astype(float)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def generate_signals(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        min_len = max(self.entry_window, self.exit_window) + 14
        if not self.validate_data(kline_df) or len(kline_df) < min_len:
            return kline_df

        df = kline_df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # 通道
        entry_high = high.rolling(self.entry_window).max()
        entry_low = low.rolling(self.entry_window).min()
        exit_high = high.rolling(self.exit_window).max()
        exit_low = low.rolling(self.exit_window).min()

        # ATR
        atr = self._calc_atr(df)

        df["entry_upper"] = entry_high
        df["entry_lower"] = entry_low
        df["exit_upper"] = exit_high
        df["exit_lower"] = exit_low
        df["atr"] = atr

        # 海龟信号逻辑
        # 突破上轨做多，跌破下轨做空；离市信号反向平仓
        prev_entry_high = entry_high.shift(1)
        prev_entry_low = entry_low.shift(1)
        prev_exit_high = exit_high.shift(1)
        prev_exit_low = exit_low.shift(1)

        conditions = [
            (close > prev_entry_high) & (close > prev_exit_high),   # 突破入市上轨
            (close < prev_entry_low) & (close < prev_exit_low),     # 突破入市下轨
            (close < prev_exit_high) & (close > prev_entry_low),    # 跌破离市上轨(多单平仓)
            (close > prev_exit_low) & (close < prev_entry_high),    # 升破离市下轨(空单平仓)
        ]
        choices = ["BUY_TURTLE", "SELL_TURTLE", "EXIT_LONG", "EXIT_SHORT"]
        action_simple = {"BUY_TURTLE": "BUY", "SELL_TURTLE": "SELL",
                         "EXIT_LONG": "SELL", "EXIT_SHORT": "BUY", "HOLD": "HOLD"}
        df["action_raw"] = np.select(conditions, choices, default="HOLD")
        df["action"] = df["action_raw"].map(action_simple).fillna("HOLD")
        last_atr = atr.iloc[-1] if len(atr) > 0 else 1
        price_position = (close - entry_low) / (entry_high - entry_low).replace(0, np.nan)
        df["strength"] = np.clip(price_position.fillna(0.5), 0, 1)

        reasons = {
            "BUY_TURTLE": f"突破{self.entry_window}日新高，海龟做多信号",
            "SELL_TURTLE": f"跌破{self.entry_window}日新低，海龟做空信号",
            "EXIT_LONG": f"跌破{self.exit_window}日高点，多单退出",
            "EXIT_SHORT": f"升破{self.exit_window}日低点，空单退出",
            "HOLD": "海龟系统持仓中或观望",
        }
        df["reason"] = df["action_raw"].map(reasons)

        # 止损位
        df["stop_loss"] = close - self.atr_mult * atr
        df["stop_profit"] = close + self.atr_mult * atr * 3  # 3倍ATR止盈

        return df

    def get_current_signal(self, kline_df: pd.DataFrame) -> Signal:
        df = self.generate_signals(kline_df)
        if df.empty:
            return Signal("HOLD", 0, "数据不足")
        last = df.iloc[-1]
        d = last.to_dict()

        stop_loss = d.get("stop_loss")
        stop_profit = d.get("stop_profit")

        return Signal(
            action=d.get("action", "HOLD"),
            strength=float(d.get("strength", 0)),
            reason=d.get("reason", ""),
            metadata={
                "entry_upper": round(float(d.get("entry_upper", 0)), 2),
                "entry_lower": round(float(d.get("entry_lower", 0)), 2),
                "atr": round(float(d.get("atr", 0)), 2) if pd.notna(d.get("atr", np.nan)) else None,
                "stop_loss": round(float(stop_loss), 2) if pd.notna(stop_loss) else None,
                "stop_profit": round(float(stop_profit), 2) if pd.notna(stop_profit) else None,
            }
        )


# ==================== 策略注册表 ====================

STRATEGY_REGISTRY = {
    "dual_ma": DualMAStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerBreakoutStrategy,
    "grid": GridTradingStrategy,
    "momentum": MomentumBreakoutStrategy,
    "turtle": TurtleStrategy,
}


def get_strategy(name: str, **params) -> BaseStrategy:
    """通过名称获取策略实例"""
    cls = STRATEGY_REGISTRY.get(name)
    if not cls:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"未知策略: {name}，可用策略: {available}")
    return cls(**params)
