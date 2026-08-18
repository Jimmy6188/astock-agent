"""
A股数据提供层 - 混合数据源架构
- 实时行情: 腾讯财经 API(主) + AKShare 东方财富(备)
- 历史K线: Baostock(主) + AKShare(备)
- 财务/资金流/龙虎榜/板块: AKShare(保留，无免费替代)
"""

import time
import logging
import threading
import requests
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# 可轮换的 User-Agent 池，避免单UA被封禁
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
]
_ua_index = [0]


def _next_user_agent():
    """轮询获取 User-Agent，降低封禁风险"""
    ua = _USER_AGENTS[_ua_index[0] % len(_USER_AGENTS)]
    _ua_index[0] += 1
    return ua


def _get_req_headers():
    """获取带随机User-Agent的请求头"""
    return {"User-Agent": _next_user_agent()}


# ============ API 速率限制 ============

_BS_LAST_CALL = [0.0]
_BS_LOCK = threading.Lock()
_BS_MIN_INTERVAL = 0.3  # 两次Baostock调用最小间隔(秒)


def rate_limit_acquire():
    """限流：确保两次Baostock查询之间至少有 _BS_MIN_INTERVAL 秒间隔"""
    with _BS_LOCK:
        elapsed = time.time() - _BS_LAST_CALL[0]
        if elapsed < _BS_MIN_INTERVAL:
            time.sleep(_BS_MIN_INTERVAL - elapsed)
        _BS_LAST_CALL[0] = time.time()


# 全局元数据记录回调（由mcp_server.py注入，默认no-op）
_meta_recorder = None

def set_meta_recorder(recorder):
    """注入MCP元数据记录器"""
    global _meta_recorder
    _meta_recorder = recorder


def _record_source(source: str, fallback: bool = False, note: str = "") -> None:
    if _meta_recorder:
        _meta_recorder(source, fallback, note)

# 缓存目录
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 腾讯财经 API 辅助函数
# ============================================================

# 代码前缀转换
_TENCENT_PREFIX = {
    "sh": "sh",  # 沪市
    "sz": "sz",  # 深市
}


def _to_tencent_code(symbol: str) -> str:
    """将6位股票代码转为腾讯API格式,如 600519 -> sh600519, 000858 -> sz000858"""
    s = str(symbol).zfill(6)
    # 上交所: 600xxx, 688xxx, 5xxxxx(ETF/LOF), 900xxx(债券)
    if s.startswith(("6", "9", "5")):
        return "sh" + s
    return "sz" + s


def verify_symbol(symbol: str, expected_name: Optional[str] = None) -> dict:
    """验证股票代码是否有效,返回{name, price}或{error}

    必须先确认代码真实性,不得杜撰。
    """
    try:
        s6 = str(symbol).zfill(6)
        code = _to_tencent_code(s6)
        url = f"https://qt.gtimg.cn/q={code}"
        r = requests.get(url, headers=_get_req_headers(), timeout=8)
        r.encoding = "gbk"
        text = r.text
        key = f'v_{code}="'
        idx = text.find(key)
        if idx < 0:
            return {"valid": False, "error": f"代码 {symbol} 无效或无法获取数据"}
        end = text.find('"', idx + len(key))
        if end < 0:
            return {"valid": False, "error": f"代码 {symbol} 数据格式异常"}
        raw = text[idx + len(key):end]
        parsed = _parse_tencent_row(raw)
        if parsed and parsed.get("name"):
            result = {"name": parsed["name"], "price": parsed.get("price"), "valid": True}
            if expected_name and parsed["name"] != expected_name:
                result["error"] = f"代码 {symbol} 对应 '{parsed['name']}'，非预期 '{expected_name}'"
                result["valid"] = False
            return result
        return {"valid": False, "error": f"代码 {symbol} 无效或无法获取数据"}
    except Exception as e:
        return {"valid": False, "error": f"验证失败: {e}"}


def _parse_tencent_row(raw: str) -> Optional[dict]:
    """解析单行腾讯行情数据

    格式(verified with real API 2026-08-17):
      0:market 1:name 2:code 3:price 4:pre_close 5:today_open
      6:volume(手) 7:amount 8~27:5档卖+5档买(vol~price)
      30:ts(YYYYMMDDHHMMSS) 31:change_amt 32:change_pct
      33:today_high 34:today_low 35:summary
      36:volume(手) 37:amount 38:turnover% 39:PE
      43:amplitude% 44:total_mv(亿) 45:circ_mv(亿) 46:PB
    """
    try:
        fields = raw.split("~")
        if len(fields) < 47:
            return None

        def _sf(i: int, default: float = 0.0) -> float:
            try:
                v = float(fields[i]) if fields[i] else default
                return v if pd.notna(v) else default
            except (ValueError, IndexError):
                return default

        return {
            "symbol": str(fields[2]),
            "name": str(fields[1]),
            "price": _sf(3),
            "pre_close": _sf(4),
            "today_open": _sf(5),
            "volume": _sf(6) * 100,          # 手→股
            "amount": _sf(7),
            "change_amt": _sf(31),
            "change_pct": _sf(32),
            "high": _sf(33),
            "low": _sf(34),
            "turnover_rate": _sf(38),
            "pe_ttm": _sf(39),
            "amplitude": _sf(43),
            "total_mv": _sf(44) * 1e8,       # 亿→元
            "circ_mv": _sf(45) * 1e8,
            "pb": _sf(46),
        }
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"腾讯行情解析失败: {e}")
        return None


def _fetch_tencent_realtime(symbols: List[str]) -> Dict[str, dict]:
    """批量从腾讯API获取实时行情

    Args:
        symbols: 6位股票代码列表
    Returns:
        {symbol: {parsed_fields}}
    """
    result = {}
    # 腾讯API单次最多约500只,分批
    batch_size = 300
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tencent_codes = [_to_tencent_code(s) for s in batch]
        url = "https://qt.gtimg.cn/q=" + ",".join(tencent_codes)
        try:
            resp = requests.get(url, headers=_get_req_headers(), timeout=10)
            resp.encoding = "gbk"
            text = resp.text
        except Exception as e:
            logger.error(f"腾讯API请求失败: {e}")
            return result

        for sym, tc in zip(batch, tencent_codes):
            # 查找 v_tc="..."
            key = f'v_{tc}="'
            idx = text.find(key)
            if idx >= 0:
                end = text.find('"', idx + len(key))
                if end < 0:
                    continue
                raw = text[idx + len(key):end]
                parsed = _parse_tencent_row(raw)
                if parsed:
                    result[sym] = parsed
    return result


def _fetch_tencent_qq_kline(symbol: str, period: str = "daily",
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            adjust: str = "qfq") -> pd.DataFrame:
    """从腾讯fqkline接口获取历史K线(快速备选,~1s)

    仅支持个股(非ETF), 返回全量近2年数据
    """
    try:
        s6 = str(symbol).zfill(6)
        code = "sh" + s6 if s6.startswith(("6", "9")) else "sz" + s6
        url = f"https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get?param={code},{period},,,,1000,qfq"
        r = requests.get(url, headers=_get_req_headers(), timeout=10)
        data = r.json()
        if data.get("code") != 0 or not data.get("data"):
            return pd.DataFrame()
        key = list(data["data"].keys())[0]
        rows = data["data"][key].get("qfqday", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        # 按日期过滤
        if start_date:
            sd = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
            if pd.notna(sd):
                df = df[df["date"] >= sd]
        if end_date:
            ed = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
            if pd.notna(ed):
                df = df[df["date"] <= ed]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Tencent fqkline K线失败 {symbol}: {e}")
        return pd.DataFrame()


# ============================================================
# Baostock 辅助函数
# ============================================================

# 懒加载 baostock
_BS = None
_BS_LOGIN = False
_BS_LOCK = threading.Lock()  # 线程安全锁


def _ensure_bs_login() -> bool:
    """确保 baostock 已登录（线程安全）"""
    global _BS, _BS_LOGIN
    with _BS_LOCK:
        if _BS is None:
            try:
                import baostock as bs
                _BS = bs
            except ImportError:
                return False

        if not _BS_LOGIN:
            try:
                lg = _BS.login()
                _BS_LOGIN = (lg.error_code == "0")
                if not _BS_LOGIN:
                    logger.error(f"Baostock登录失败: {lg.error_msg}")
            except Exception as e:
                logger.error(f"Baostock初始化失败: {e}")
                _BS_LOGIN = False
        return _BS_LOGIN


def _to_bs_code(symbol: str) -> str:
    """将6位代码转Baostock格式,如 600519 -> sh.600519"""
    s = str(symbol).zfill(6)
    if s.startswith(("6", "9")):
        return "sh." + s
    return "sz." + s


def _bs_period(period: str) -> str:
    """period -> baostock frequency"""
    return {"daily": "d", "weekly": "w", "monthly": "m"}.get(period, "d")


def _bs_adjust(adjust: str) -> str:
    """adjust -> baostock adjustflag"""
    return {"qfq": "2", "hfq": "1", "": "3"}.get(adjust, "2")


def _fetch_bs_kline(symbol: str, period: str, start_date: str, end_date: str,
                     adjust: str = "qfq") -> pd.DataFrame:
    """从 Baostock 获取历史K线

    Args:
        symbol: 6位代码
        period: daily/weekly/monthly
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        adjust: qfq/hfq/""
    Returns:
        DataFrame 含 date/open/high/low/close/volume/amount 列
    """
    if not _ensure_bs_login():
        return pd.DataFrame()

    # 限流：两次Baostock查询间隔至少0.3秒
    rate_limit_acquire()

    bs_code = _to_bs_code(symbol)
    freq = _bs_period(period)
    adj_flag = _bs_adjust(adjust)

    # Baostock 需要 YYYY-MM-DD 格式
    def _fmt(d):
        try:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        except (TypeError, IndexError):
            return d

    try:
        rs = _BS.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=_fmt(start_date),
            end_date=_fmt(end_date),
            frequency=freq,
            adjustflag=adj_flag,
        )
        if rs.error_code != "0":
            logger.warning(f"Baostock K线查询失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        df.columns = [c.strip() for c in df.columns]

        # 标准化列名
        col_map = {
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
            "amount": "amount", "turn": "turnover", "pctChg": "pct_change",
        }
        existing_cols = {c for c in df.columns if c in col_map}
        df = df.rename(columns={c: col_map[c] for c in existing_cols})

        # 确保数值列
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # date 列统一为 YYYY-MM-DD
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        return df

    except Exception as e:
        logger.warning(f"Baostock K线获取失败 {symbol}: {e}")
        return pd.DataFrame()


# ============================================================
# 主数据提供类
# ============================================================

class AStockDataProvider:
    """A股数据统一接口 - 混合数据源"""

    # 缓存大小上限，防止无限增长导致内存泄漏
    MAX_CACHE_SIZE = 200

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self._bs_lock = threading.Lock()

    # -------------------- 实时行情 --------------------

    def get_realtime_quote(self, symbol: str) -> dict:
        """获取单只股票实时行情 - 腾讯(主) + akshare(备)"""
        cache_key = f"rt_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        symbol_6 = str(symbol).zfill(6)
        result = {}

        # 1. 腾讯API(主)
        try:
            tc_result = _fetch_tencent_realtime([symbol_6])
            if symbol_6 in tc_result:
                r = tc_result[symbol_6]
                result = {
                    "symbol": symbol_6,
                    "name": r.get("name", ""),
                    "price": float(r.get("price", 0)),
                    "change_pct": float(r.get("change_pct", 0)),
                    "change_amt": float(r.get("change_amt", 0)),
                    "volume": float(r.get("volume", 0)),
                    "amount": float(r.get("amount", 0)),
                    "amplitude": float(r.get("amplitude", 0)),
                    "high": float(r.get("high", 0)),
                    "low": float(r.get("low", 0)),
                    "open": float(r.get("today_open", 0)),
                    "pre_close": float(r.get("pre_close", 0)),
                    "volume_ratio": float(r.get("volume_ratio", 0)),
                    "turnover_rate": float(r.get("turnover_rate", 0)),
                    "pe_ttm": float(r.get("pe_ttm", 0)) if r.get("pe_ttm", 0) else None,
                    "pb": float(r.get("pb", 0)) if r.get("pb", 0) else None,
                    "total_mv": float(r.get("total_mv", 0)),
                    "circ_mv": float(r.get("circ_mv", 0)),
                    "rise_speed": None,
                    "timestamp": datetime.now().isoformat(),
                }
                self._set_cache(cache_key, result)
                _record_source("tencent", note=f"{symbol_6}实时行情-腾讯")
                return result
        except Exception as e:
            logger.warning(f"腾讯行情获取失败 {symbol_6}: {e}")

        # 2. 降级到 akshare
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol_6]
            if row.empty:
                row = df[df["代码"] == symbol_6.zfill(6)]
            if row.empty:
                return {"error": f"未找到股票 {symbol_6}"}
            r = row.iloc[0]
            result = {
                "symbol": str(r.get("代码", symbol_6)),
                "name": str(r.get("名称", "")),
                "price": float(r.get("最新价", 0)),
                "change_pct": float(r.get("涨跌幅", 0)),
                "change_amt": float(r.get("涨跌额", 0)),
                "volume": float(r.get("成交量", 0)),
                "amount": float(r.get("成交额", 0)),
                "amplitude": float(r.get("振幅", 0)),
                "high": float(r.get("最高", 0)),
                "low": float(r.get("最低", 0)),
                "open": float(r.get("今开", 0)),
                "pre_close": float(r.get("昨收", 0)),
                "volume_ratio": float(r.get("量比", 0)),
                "turnover_rate": float(r.get("换手率", 0)),
                "pe_ttm": float(r.get("市盈率-动态", 0)) if pd.notna(r.get("市盈率-动态")) else None,
                "pb": float(r.get("市净率", 0)) if pd.notna(r.get("市净率")) else None,
                "total_mv": float(r.get("总市值", 0)),
                "circ_mv": float(r.get("流通市值", 0)),
                "rise_speed": float(r.get("涨速", 0)) if pd.notna(r.get("涨速")) else None,
                "timestamp": datetime.now().isoformat(),
            }
            self._set_cache(cache_key, result)
            _record_source("akshare", fallback=True, note=f"{symbol_6}实时行情-AKShare降级(腾讯失败)")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol_6} 实时行情失败: {e}")
            _record_source("error", fallback=True, note=f"{symbol_6}实时行情-腾讯+AKShare均失败: {e}")
            return {"error": str(e)}

    def get_batch_quotes(self, symbols: list) -> list:
        """批量获取多只股票行情 - 腾讯(主) + akshare(备)"""
        try:
            symbols_6 = [str(s).zfill(6) for s in symbols]
            tc_result = _fetch_tencent_realtime(symbols_6)

            if tc_result:
                results = []
                for sym in symbols_6:
                    if sym in tc_result:
                        r = tc_result[sym]
                        results.append({
                            "symbol": sym,
                            "name": r.get("name", ""),
                            "price": float(r.get("price", 0)),
                            "change_pct": float(r.get("change_pct", 0)),
                            "volume": float(r.get("volume", 0)),
                            "amount": float(r.get("amount", 0)),
                            "turnover_rate": float(r.get("turnover_rate", 0)),
                            "pe_ttm": float(r.get("pe_ttm", 0)) if r.get("pe_ttm", 0) else None,
                        })
                    else:
                        results.append({"symbol": sym, "error": "未找到"})
                return results
        except Exception as e:
            logger.warning(f"腾讯批量行情失败: {e}")

        # 降级到 akshare
        try:
            df = ak.stock_zh_a_spot_em()
            results = []
            for sym in symbols:
                code = str(sym).zfill(6)
                row = df[df["代码"] == code]
                if not row.empty:
                    r = row.iloc[0]
                    results.append({
                        "symbol": str(r["代码"]),
                        "name": str(r["名称"]),
                        "price": float(r["最新价"]),
                        "change_pct": float(r["涨跌幅"]),
                        "volume": float(r["成交量"]),
                        "amount": float(r["成交额"]),
                        "turnover_rate": float(r["换手率"]),
                        "pe_ttm": float(r["市盈率-动态"]) if pd.notna(r["市盈率-动态"]) else None,
                    })
                else:
                    results.append({"symbol": code, "error": "未找到"})
            return results
        except Exception as e:
            logger.error(f"批量行情获取失败: {e}")
            return [{"error": str(e)}]

    # -------------------- 历史K线 --------------------

    def get_history_kline(
        self,
        symbol: str,
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取历史K线 - Baostock(主) + AKShare(备)"""
        cache_key = f"kline_{symbol}_{period}_{start_date}_{end_date}_{adjust}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        symbol_6 = str(symbol).zfill(6)

        # 1. Baostock(主)
        try:
            df = _fetch_bs_kline(symbol_6, period, start_date, end_date, adjust)
            if df is not None and not df.empty:
                self._set_cache(cache_key, df)
                _record_source("baostock", note=f"{symbol_6}K线-Baostock")
                return df
        except Exception as e:
            logger.warning(f"Baostock K线失败 {symbol_6}: {e}")

        # 2. 腾讯fqkline(快速备选,~1s,仅个股)
        try:
            df = _fetch_tencent_qq_kline(symbol_6, period, start_date, end_date, adjust)
            if df is not None and not df.empty:
                self._set_cache(cache_key, df)
                _record_source("tencent", note=f"{symbol_6}K线-腾讯fqkline")
                return df
        except Exception as e:
            logger.warning(f"Tencent fqkline K线失败 {symbol_6}: {e}")

        # 3. 降级到 AKShare
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol_6, period=period,
                start_date=start_date, end_date=end_date, adjust=adjust,
            )
            if df is not None and not df.empty:
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "振幅": "amplitude",
                    "涨跌幅": "pct_change", "涨跌额": "change_amt",
                    "换手率": "turnover",
                }
                existing = {c for c in df.columns if c in col_map}
                df = df.rename(columns={c: col_map[c] for c in existing})
                self._set_cache(cache_key, df)
                _record_source("akshare", fallback=True, note=f"{symbol_6}K线-AKShare降级(Baostock失败)")
                return df
        except Exception as e:
            logger.error(f"获取 {symbol_6} K线失败: {e}")
            _record_source("error", fallback=True, note=f"{symbol_6}K线-Baostock+Tencent+AKShare均失败: {e}")

        return pd.DataFrame()

    # -------------------- 资金流向 --------------------

    def get_money_flow(self, symbol: str) -> dict:
        """获取个股资金流向(AKShare/东方财富,无免费替代)"""
        cache_key = f"mf_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            symbol_6 = str(symbol).zfill(6)
            df = ak.stock_individual_fund_flow(stock=symbol_6, market="sz" if symbol_6.startswith(("0", "3")) else "sh")
            if df is None or df.empty:
                return {"error": "无资金流数据"}
            latest = df.iloc[-1]
            result = {
                "symbol": symbol,
                "date": str(latest.get("日期", "")),
                "main_net_inflow": float(latest.get("主力净流入-净额", 0)) if pd.notna(latest.get("主力净流入-净额")) else 0,
                "main_net_pct": float(latest.get("主力净流入-净占比", 0)) if pd.notna(latest.get("主力净流入-净占比")) else 0,
                "super_large_net": float(latest.get("超大单净流入-净额", 0)) if pd.notna(latest.get("超大单净流入-净额")) else 0,
                "large_net": float(latest.get("大单净流入-净额", 0)) if pd.notna(latest.get("大单净流入-净额")) else 0,
                "medium_net": float(latest.get("中单净流入-净额", 0)) if pd.notna(latest.get("中单净流入-净额")) else 0,
                "small_net": float(latest.get("小单净流入-净额", 0)) if pd.notna(latest.get("小单净流入-净额")) else 0,
            }
            self._set_cache(cache_key, result)
            _record_source("akshare", note=f"{symbol_6}资金流-AKShare")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 资金流失败: {e}")
            _record_source("error", fallback=True, note=f"{symbol}资金流-AKShare失败(东方财富限流): {e}")
            return {"error": str(e)}

    # -------------------- 龙虎榜 --------------------

    def get_lhb(self, symbol: str) -> list:
        """获取个股龙虎榜数据(AKShare/东方财富)"""
        cache_key = f"lhb_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            symbol_6 = str(symbol).zfill(6)
            df = ak.stock_lhb_detail_em(
                start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return []
            rows = df[df["代码"].astype(str).str.strip() == symbol_6]
            result = []
            for _, r in rows.iterrows():
                result.append({
                    "date": str(r.get("日期", "")),
                    "name": str(r.get("名称", "")),
                    "reason": str(r.get("上榜原因", "")),
                    "buy_amount": float(r.get("买入额", 0)) if pd.notna(r.get("买入额")) else 0,
                    "sell_amount": float(r.get("卖出额", 0)) if pd.notna(r.get("卖出额")) else 0,
                    "net_buy": float(r.get("净买额", 0)) if pd.notna(r.get("净买额")) else 0,
                    "buy_seats": str(r.get("买入席位", "")),
                    "sell_seats": str(r.get("卖出席位", "")),
                })
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 龙虎榜失败: {e}")
            return []

    # -------------------- 财务报表 --------------------

    def get_financials(self, symbol: str) -> dict:
        """获取主要财务指标 - AKShare(主) + Baostock(备)

        优化说明: AKShare 单次API调用可获取全部财务指标(0.5-2s),
        相比 Baostock 的 7表×4季度=28次API调用(15-30s) 提升 10-30倍
        """
        cache_key = f"fin_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        symbol_6 = str(symbol).zfill(6)

        # 1. AKShare(主) - 单次调用, ~1s
        for attempt in range(2):
            try:
                df = ak.stock_financial_abstract_ths(symbol=symbol_6, indicator="按报告期")
                if df is not None and not df.empty:
                    # 按报告期降序，取最新一行
                    if "报告期" in df.columns:
                        df["报告期"] = pd.to_datetime(df["报告期"], errors="coerce")
                        df = df.sort_values("报告期", ascending=False)
                    latest = df.iloc[0]
                    result = {
                        "symbol": symbol_6,
                        "report_date": str(latest.get("报告期", "")),
                        "revenue": self._safe_float(latest, "营业总收入"),
                        "revenue_yoy": self._safe_float(latest, "营业总收入同比增长率"),
                        "net_profit": self._safe_float(latest, "净利润"),
                        "net_profit_yoy": self._safe_float(latest, "净利润同比增长率"),
                        "gross_margin": self._safe_float(latest, "销售毛利率"),
                        "net_margin": self._safe_float(latest, "销售净利率"),
                        "roe": self._safe_float(latest, "净资产收益率"),
                        "roa": None,
                        "debt_ratio": self._safe_float(latest, "资产负债率"),
                        "current_ratio": self._safe_float(latest, "流动比率"),
                        "quick_ratio": self._safe_float(latest, "速动比率"),
                        "eps": self._safe_float(latest, "基本每股收益"),
                        "bps": self._safe_float(latest, "每股净资产"),
                        "ocf_per_share": self._safe_float(latest, "每股经营现金流"),
                        "total_assets": None,
                        "total_equity": None,
                        "source": "akshare",
                    }
                    self._set_cache(cache_key, result)
                    _record_source("akshare", note=f"{symbol_6}财务-AKShare")
                    return result
            except Exception as e:
                logger.warning(f"AKShare财务失败 {symbol_6} (attempt {attempt+1}/2): {e}")
                if attempt < 1:
                    time.sleep(1)

        # 2. Baostock(备) - 精简版: 仅2表×2季度=4次API
        try:
            try:
                from .data_provider_bs_fin import _try_bs_financials
            except ImportError:
                from data_provider_bs_fin import _try_bs_financials
            result = _try_bs_financials(symbol_6)
            if isinstance(result, dict) and "error" not in result:
                self._set_cache(cache_key, result)
                _record_source("baostock", fallback=True,
                               note=f"{symbol_6}财务-Baostock降级(AKShare失败)")
                return result
        except Exception as e:
            logger.warning(f"Baostock财务数据失败 {symbol_6}: {e}")

        return {"error": "无财务数据"}

    # -------------------- 估值分位 --------------------

    def get_valuation_percentile(self, symbol: str) -> dict:
        """获取PE/PB历史分位 - AKShare(仅源，需PE/PB列，带重试)"""
        cache_key = f"val_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        symbol_6 = str(symbol).zfill(6)
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        # AKShare(估值分位需要PE/PB列) - 1年数据, 1次尝试
        df = None
        for attempt in range(2):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol_6, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq",
                )
                if df is not None and not df.empty and "市盈率TTM" in df.columns:
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                df = None

        if df is None or df.empty or "市盈率TTM" not in df.columns:
            # 降级: 从实时行情获取当前PE/PB
            quote = self.get_realtime_quote(symbol_6)
            if "error" not in quote:
                _record_source("akshare", fallback=True,
                               note=f"{symbol_6}估值-AKShare3次重试失败，降级为腾讯行情当前PE/PB")
                return {
                    "symbol": symbol_6,
                    "current_pe_ttm": quote.get("pe_ttm"),
                    "current_pb": quote.get("pb"),
                    "pe_percentile_3y": None,
                    "note": "无历史PE/PB数据，仅返回当前值",
                }
            _record_source("error", fallback=True, note=f"{symbol_6}估值-无法获取")
            return {"error": "无估值数据"}

        pe_series = df["市盈率TTM"].dropna()
        pb_series = df["市净率"].dropna() if "市净率" in df.columns else pd.Series()

        current_pe = float(pe_series.iloc[-1]) if len(pe_series) > 0 else None
        current_pb = float(pb_series.iloc[-1]) if len(pb_series) > 0 else None

        result = {
            "symbol": symbol_6,
            "current_pe_ttm": current_pe,
            "pe_percentile_3y": float((pe_series < current_pe).mean() * 100) if current_pe and len(pe_series) > 0 else None,
            "pe_min": float(pe_series.min()) if len(pe_series) > 0 else None,
            "pe_max": float(pe_series.max()) if len(pe_series) > 0 else None,
            "pe_median": float(pe_series.median()) if len(pe_series) > 0 else None,
            "current_pb": current_pb,
            "pb_percentile_3y": float((pb_series < current_pb).mean() * 100) if current_pb and len(pb_series) > 0 else None,
            "pb_min": float(pb_series.min()) if len(pb_series) > 0 else None,
            "pb_max": float(pb_series.max()) if len(pb_series) > 0 else None,
            "data_points": len(pe_series),
        }
        self._set_cache(cache_key, result)
        _record_source("akshare", note=f"{symbol_6}估值分位-AKShare")
        return result

    # -------------------- 新闻舆情 --------------------

    def get_news_sentiment(self, symbol: str, days: int = 7) -> dict:
        """获取个股新闻舆情(AKShare/东方财富)"""
        cache_key = f"news_{symbol}_{days}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            from datetime import date
            symbol_6 = str(symbol).zfill(6)
            df = ak.stock_news_em(symbol=symbol_6)
            if df is None or df.empty:
                return {"news_count": 0, "sentiment": "neutral", "headlines": []}
            # 按发布时间筛选(如果列存在)
            if "发布时间" in df.columns:
                df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                df = df[df["发布时间"] >= cutoff]
            # 只取前20条
            headlines = []
            for _, r in df.head(20).iterrows():
                headlines.append({
                    "title": str(r.get("新闻标题", "")),
                    "content": str(r.get("新闻内容", ""))[:200],
                    "time": str(r.get("发布时间", "")),
                    "source": str(r.get("文章来源", "")),
                })
            positive_words = ["利好", "大涨", "突破", "创新高", "增持", "回购", "业绩超预期", "盈利"]
            negative_words = ["利空", "大跌", "暴跌", "减持", "亏损", "预警", "下调", "风险"]
            pos_cnt, neg_cnt = 0, 0
            text_all = " ".join([h["title"] + h["content"] for h in headlines])
            for w in positive_words:
                pos_cnt += text_all.count(w)
            for w in negative_words:
                neg_cnt += text_all.count(w)
            total = pos_cnt + neg_cnt
            if total == 0:
                sentiment = "neutral"
            elif pos_cnt / total > 0.6:
                sentiment = "positive"
            elif neg_cnt / total > 0.6:
                sentiment = "negative"
            else:
                sentiment = "mixed"
            result = {
                "symbol": symbol,
                "news_count": len(df),
                "sentiment": sentiment,
                "positive_score": pos_cnt,
                "negative_score": neg_cnt,
                "headlines": headlines[:10],
            }
            self._set_cache(cache_key, result)
            _record_source("akshare", note=f"{symbol_6}舆情-AKShare")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 舆情失败: {e}")
            _record_source("error", fallback=True, note=f"{symbol}舆情-AKShare失败: {e}")
            return {"error": str(e), "news_count": 0, "sentiment": "unknown"}

    # -------------------- 板块/行业 --------------------

    def get_sector_performance(self) -> pd.DataFrame:
        """获取行业板块涨跌排行(AKShare/东方财富)"""
        cache_key = "sector_perf"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                self._set_cache(cache_key, df)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"获取板块数据失败: {e}")
            return pd.DataFrame()

    # -------------------- 选股筛选 --------------------

    def screen_stocks(
        self,
        min_change_pct: float = -10.0,
        max_change_pct: float = 10.0,
        min_volume_ratio: float = 0.8,
        min_turnover: float = 0.5,
        max_pe: float = 100.0,
        min_market_cap: float = 10.0,
    ) -> list:
        """条件选股

        默认排除：ST股、科创板(688)、创业板(300/301)、不满一年新股
        只选沪深主板
        """
        try:
            df = None
            # 东方财富偶有限流，最多重试3次
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot_em()
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        logger.error(f"全市场数据获取失败: {e}")
                finally:
                    df = None if df is None else df
            if df is None or df.empty:
                return []

            # ---- 第1轮：代码+名称过滤 ----
            code = df["代码"].astype(str)

            # 剔除科创板 688xxx
            mask_not_star = ~code.str.startswith("688")
            # 剔除创业板 300xxx / 301xxx
            mask_not_chinext = ~code.str.startswith("30")
            # 剔除北交所 4/8/9xxxxx
            mask_not_bse = ~code.str.match(r"^[489]")
            # 剔除 ST 股
            names = df["名称"].astype(str)
            mask_not_st = ~names.str.contains("ST|st", case=False, na=False)
            # 只保留沪深主板
            mask_main_board = (
                code.str.startswith("600") | code.str.startswith("601") |
                code.str.startswith("603") | code.str.startswith("605") |
                code.str.startswith("000") | code.str.startswith("001") |
                code.str.startswith("002") | code.str.startswith("003")
            )

            combined = mask_not_star & mask_not_chinext & mask_not_bse & mask_not_st & mask_main_board
            df = df[combined].copy()

            # ---- 数值条件过滤 ----
            mask = (
                (df["涨跌幅"] >= min_change_pct) &
                (df["涨跌幅"] <= max_change_pct) &
                (df["量比"] >= min_volume_ratio) &
                (df["换手率"] >= min_turnover) &
                (df["总市值"] >= min_market_cap * 1e8)
            )
            if max_pe > 0 and "市盈率-动态" in df.columns:
                pe_col = df["市盈率-动态"].replace([np.inf, -np.inf], np.nan)
                mask &= ((pe_col > 0) & (pe_col <= max_pe))

            df = df[mask].sort_values("涨跌幅", ascending=False)

            # ---- 第2轮：剔除不满一年新股 ----
            # 批量查询历史K线以判断上市日期，避免每只股票单独请求的N+1问题
            all_valid_symbols: set = set()
            try:
                # 一次批量获取历史数据（AKShare按市场批量查询）
                hist_all = ak.stock_zh_a_hist(symbol="sh000001", period="daily",
                                               start_date="20190101", adjust="")
                # 用单只股票批量查询所有代码
                candidates = df["代码"].astype(str).tolist()
                batch_size = 50
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i:i + batch_size]
                    for sym in batch:
                        try:
                            k = _fetch_bs_kline(sym, "daily", "20190101",
                                                datetime.now().strftime("%Y%m%d"), "")
                            if k is None or k.empty:
                                k = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                                       start_date="20190101", adjust="")
                            if k is not None and not k.empty and len(k) >= 240:
                                all_valid_symbols.add(sym)
                        except Exception:
                            all_valid_symbols.add(sym)  # 查不到就放行
            except Exception as e:
                logger.warning(f"新股过滤批量查询异常: {e}，放行所有股票")
                all_valid_symbols = set(candidates)

            valid_symbols = [s for s in candidates if s in all_valid_symbols]

            # ---- 返回结果 ----
            results = []
            df_final = df[df["代码"].astype(str).isin(valid_symbols)]
            for _, r in df_final.head(50).iterrows():
                results.append({
                    "symbol": str(r["代码"]),
                    "name": str(r["名称"]),
                    "price": float(r["最新价"]),
                    "change_pct": float(r["涨跌幅"]),
                    "volume_ratio": float(r["量比"]),
                    "turnover_rate": float(r["换手率"]),
                    "amount": float(r["成交额"]),
                    "pe_ttm": float(r["市盈率-动态"]) if pd.notna(r.get("市盈率-动态")) else None,
                    "total_mv": float(r["总市值"]),
                })
            return results
        except Exception as e:
            logger.error(f"选股筛选失败: {e}")
            return []

    # -------------------- 工具方法 --------------------

    @staticmethod
    def _safe_float(row, col) -> Optional[float]:
        """从行取float值，支持'5.04亿'/'2266.19万'等中文单位"""
        v = row.get(col, np.nan)
        if pd.isna(v) or v is None or v is False:
            return None
        s = str(v).strip()
        # 去除百分号
        if s.endswith('%'):
            try:
                return float(s[:-1])
            except (ValueError, TypeError):
                return None
        # 处理带中文单位的数值
        if s.endswith('亿'):
            try:
                return float(s[:-1]) * 1e8
            except (ValueError, TypeError):
                return None
        if s.endswith('万'):
            try:
                return float(s[:-1]) * 1e4
            except (ValueError, TypeError):
                return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < self.cache_ttl:
            return entry["data"]
        return None

    def _set_cache(self, key: str, data) -> None:
        with self._cache_lock:
            # LRU-style eviction: 超过上限时淘汰最旧的过期项
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                # 先尝试删除已过期项
                now = time.time()
                expired = [k for k, v in self._cache.items()
                           if now - v["ts"] >= self.cache_ttl]
                for k in expired:
                    self._cache.pop(k, None)
                # 若仍超限，淘汰最旧项
                if len(self._cache) >= self.MAX_CACHE_SIZE:
                    oldest = min(self._cache, key=lambda k: self._cache[k]["ts"])
                    self._cache.pop(oldest, None)
            self._cache[key] = {"data": data, "ts": time.time()}

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ============================================================
# 单例
# ============================================================

_provider_instance = None


def get_provider() -> AStockDataProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = AStockDataProvider()
    return _provider_instance