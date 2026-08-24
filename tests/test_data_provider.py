"""腾讯行情解析器单测：字段映射与异常输入"""
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_provider import _parse_tencent_row, _to_tencent_code  # noqa: E402


SAMPLE_HEAD = [
    "1",                # 0 market
    "贵州茅台",          # 1 name
    "600519",           # 2 code
    "1500.00",          # 3 price
    "1490.00",          # 4 pre_close
    "1495.00",          # 5 open
    "20000",            # 6 volume(手)
    "3000000000",       # 7 amount
]

# 按索引显式构造 8~46，避免手数错位
_TAIL = {30: "20260824150000", 31: "10.00", 32: "0.67",
         33: "1510.00", 34: "1488.00", 38: "0.15", 39: "25.30",
         43: "1.50", 44: "18800", 45: "18800", 46: "8.50"}
SAMPLE_ROW = "~".join(SAMPLE_HEAD + [_TAIL.get(i, "") for i in range(8, 47)])


class TestTencentParse(unittest.TestCase):

    def test_parse_fields(self):
        row = _parse_tencent_row(SAMPLE_ROW)
        self.assertIsNotNone(row)
        self.assertEqual(row["symbol"], "600519")
        self.assertEqual(row["name"], "贵州茅台")
        self.assertAlmostEqual(row["price"], 1500.00)
        self.assertAlmostEqual(row["volume"], 20000 * 100)      # 手→股
        self.assertAlmostEqual(row["total_mv"], 18800 * 1e8)    # 亿→元

    def test_too_short_returns_none(self):
        self.assertIsNone(_parse_tencent_row("1~贵州茅台~600519"))

    def test_code_prefix(self):
        self.assertEqual(_to_tencent_code("600519"), "sh600519")
        self.assertEqual(_to_tencent_code("688001"), "sh688001")
        self.assertEqual(_to_tencent_code("000858"), "sz000858")
        self.assertEqual(_to_tencent_code("300750"), "sz300750")
        self.assertEqual(_to_tencent_code("512880"), "sh512880")  # ETF


if __name__ == "__main__":
    unittest.main()
