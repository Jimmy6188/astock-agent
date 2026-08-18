"""
Baostock 财务数据辅助函数
用于 get_financials 的主数据源，AKShare 为备降
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 复用已有的 data_provider 模块
try:
    from . import data_provider as dp
except ImportError:
    import data_provider as dp


def _to_bs_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    return ("sh." + s) if s.startswith(("6", "9")) else ("sz." + s)


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return None


def _fetch_bs_df(bs_func, **kwargs) -> pd.DataFrame:
    """通用Baostock数据获取包装"""
    if not dp._ensure_bs_login():
        return pd.DataFrame()
    try:
        rs = bs_func(**kwargs)
        if rs.error_code != "0":
            return pd.DataFrame()
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=rs.fields)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _g(df, col, default=None):
    """从DataFrame取第一行某列"""
    if df is None or df.empty or col not in df.columns:
        return default
    try:
        v = df.iloc[0][col]
        return default if pd.isna(v) else v
    except Exception:
        return default


# ============================================================
# get_financials 的 Baostock 实现
# ============================================================

def _try_bs_financials(symbol_6: str) -> dict:
    """从 Baostock 获取最新季度财务数据

    返回格式与 AKShare 财务摘要一致(分析器兼容)
    百分比字段已转为百分比数值(如 0.123 → 12.3)
    """
    import baostock as bs
    bs_code = _to_bs_code(symbol_6)

    # 按季度尝试: 2026Q2 → 2026Q1 → 2025Q4 → 2025Q3
    profit = balance = cashflow = growth = dupont = operation = expr = pd.DataFrame()
    for year, quarter in [(2026, 2), (2026, 1), (2025, 4), (2025, 3)]:
        profit = _fetch_bs_df(bs.query_profit_data, code=bs_code, year=year, quarter=quarter)
        balance = _fetch_bs_df(bs.query_balance_data, code=bs_code, year=year, quarter=quarter)
        cashflow = _fetch_bs_df(bs.query_cash_flow_data, code=bs_code, year=year, quarter=quarter)
        growth = _fetch_bs_df(bs.query_growth_data, code=bs_code, year=year, quarter=quarter)
        dupont = _fetch_bs_df(bs.query_dupont_data, code=bs_code, year=year, quarter=quarter)
        operation = _fetch_bs_df(bs.query_operation_data, code=bs_code, year=year, quarter=quarter)
        end_m = f"{year}-{quarter * 3:02d}-30"
        expr = _fetch_bs_df(bs.query_performance_express_report, code=bs_code,
                             start_date=f"{year}-01-01", end_date=end_m)
        if not (profit.empty and balance.empty and cashflow.empty
                and growth.empty and dupont.empty and operation.empty and expr.empty):
            break
    else:
        return {"error": "Baostock无财务数据"}

    result = {"symbol": symbol_6, "source": "baostock"}

    # ---- 利润表 ----
    if not profit.empty:
        result["revenue"] = _safe_float(_g(profit, "MBRevenue"))
        result["net_profit"] = _safe_float(_g(profit, "netProfit"))
        result["eps"] = _safe_float(_g(profit, "epsTTM"))
        roe_raw = _safe_float(_g(profit, "roeAvg"))
        result["roe"] = roe_raw * 100 if roe_raw is not None else None
        np_m = _safe_float(_g(profit, "npMargin"))
        result["net_margin"] = np_m * 100 if np_m is not None else None
        gp_m = _safe_float(_g(profit, "gpMargin"))
        result["gross_margin"] = gp_m * 100 if gp_m is not None else None
        result["report_date"] = str(_g(profit, "statDate", ""))
        result["pub_date"] = str(_g(profit, "pubDate", ""))

    # ---- 资产负债表 ----
    if not balance.empty:
        result["current_ratio"] = _safe_float(_g(balance, "currentRatio"))
        result["quick_ratio"] = _safe_float(_g(balance, "quickRatio"))
        dl = _safe_float(_g(balance, "liabilityToAsset"))
        result["debt_ratio"] = dl * 100 if dl is not None else None

    # ---- 现金流表 ----
    if not cashflow.empty:
        cfo_np = _safe_float(_g(cashflow, "CFOToNP"))
        eps = result.get("eps")
        if cfo_np is not None and eps is not None:
            result["ocf_per_share"] = cfo_np * eps
        else:
            result["ocf_per_share"] = None

    # ---- 成长能力 ----
    rev_yoy = None
    if not growth.empty:
        ni_yoy = _safe_float(_g(growth, "YOYNI"))
        eq_yoy = _safe_float(_g(growth, "YOYEquity"))
        asset_yoy = _safe_float(_g(growth, "YOYAsset"))
        result["net_profit_yoy"] = ni_yoy * 100 if ni_yoy is not None else None
        result["equity_yoy"] = eq_yoy * 100 if eq_yoy is not None else None
        result["asset_yoy"] = asset_yoy * 100 if asset_yoy is not None else None
        rev_yoy = _safe_float(_g(growth, "YOYPNI"))

    # ---- 营收同比增长(预披露, 若成长表无) ----
    if not expr.empty:
        gryoy = _safe_float(_g(expr, "performanceExpressGRYOY"))
        result["revenue_yoy"] = gryoy * 100 if gryoy is not None else (rev_yoy * 100 if rev_yoy is not None else None)
    else:
        result["revenue_yoy"] = rev_yoy * 100 if rev_yoy is not None else None

    # ---- 杜邦分析 ----
    if not dupont.empty:
        d_roe = _safe_float(_g(dupont, "dupontROE"))
        d_nitogr = _safe_float(_g(dupont, "dupontNitogr"))
        d_at = _safe_float(_g(dupont, "dupontAssetTurn"))
        result["dupont_roe"] = d_roe * 100 if d_roe is not None else None
        if d_nitogr is not None and d_at is not None:
            result["roa"] = d_nitogr * d_at * 100

    # ---- 营运能力 ----
    if not operation.empty:
        result["nr_turn_ratio"] = _safe_float(_g(operation, "NRTurnRatio"))
        result["inv_turn_ratio"] = _safe_float(_g(operation, "INVTurnRatio"))
        if "roa" not in result or result.get("roa") is None:
            at = _safe_float(_g(operation, "AssetTurnRatio"))
            if at is not None:
                result["asset_turn_ratio"] = at

    return result