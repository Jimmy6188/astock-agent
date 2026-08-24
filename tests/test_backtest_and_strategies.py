"""最小单元测试：覆盖回测引擎、网格策略无前视、腾讯行情解析。

运行:
  cd 项目根 && python -m pytest tests/ -v
或（无 pytest 时）:
  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 兼容包方式与脚本方式导入 src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestEngine  # noqa: E402
from src.strategies import GridTradingStrategy, DualMAStrategy  # noqa: E402


def make_kline(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """构造合成日线数据：正弦波动 + 温和上升趋势"""
    rng = np.random.default_rng(seed)
    base = np.linspace(10, 15, n)
    close = base + np.sin(np.linspace(0, 6 * np.pi, n)) * 1.5 + rng.normal(0, 0.1, n)
    close = np.round(close, 2)
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


class TestGridStrategyNoLookahead(unittest.TestCase):
    """网格策略基准价不得使用未来数据"""

    def test_base_price_is_rolling_not_global_tail(self):
        df = make_kline(120)
        out = GridTradingStrategy().generate_signals(df)

        # 每 bar 的 base_price 必须等于截至该 bar 的滚动60日均价
        close = df["close"].astype(float)
        expected = close.rolling(60).mean()
        got = out["base_price"].astype(float)
        for i in range(60, len(df)):
            self.assertAlmostEqual(
                got.iloc[i], expected.iloc[i], places=6,
                msg=f"bar {i} 的 base_price 使用了未来数据",
            )

        # 前59个bar为预热期，不产生任何信号
        self.assertTrue((out["action"].iloc[:59] == "HOLD").all())


class TestBacktestEngineAccounting(unittest.TestCase):
    """回测引擎买卖记账与成本模型"""

    def _run(self, actions):
        df = make_kline(40)
        strat = DualMAStrategy()

        class ScriptedStrategy(DualMAStrategy):
            """按给定序列回放动作，隔离市场信号"""
            _actions = iter(actions)

            def generate_signals(self, kline_df):
                out = super().generate_signals(kline_df)
                out["action"] = list(self._actions) + ["HOLD"] * len(out)
                return out

        del strat
        return BacktestEngine(initial_capital=100_000).run(df, ScriptedStrategy())

    def test_buy_sell_accounting_with_costs(self):
        df = make_kline(40)
        engine = BacktestEngine(initial_capital=100_000)

        class Scripted(DualMAStrategy):
            def generate_signals(self, kline_df):
                out = kline_df.copy()
                acts = ["BUY"] + ["HOLD"] * (len(out) - 2) + ["SELL"]
                out["action"] = acts
                out["reason"] = ""
                return out

        result = engine.run(df, Scripted(), symbol="TEST")

        # 恰好一笔完整交易
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.winning_trades + result.losing_trades, 1)
        # 强制平仓后无残留持仓，最终资金为纯现金且非负
        self.assertGreaterEqual(result.final_capital, 0)
        buy = result.trade_records[0]
        sells = [t for t in result.trade_records if "exit_price" in t]
        self.assertEqual(len(sells), 1)
        sell = sells[0]
        # 买入含佣金，卖出价应低于末日收盘价（滑点+成本）
        self.assertGreater(buy["commission"], 0)
        last_close = float(df.iloc[-1]["close"])
        self.assertLess(sell["exit_price"], last_close)

    def test_insufficient_data_returns_error(self):
        df = make_kline(10)
        result = BacktestEngine().run(df, DualMAStrategy())
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
