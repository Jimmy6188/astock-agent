"""
深度价值投资分析引擎 (Value Investing Deep Dive)
灵感来源: ai-berkshire (15.6k stars) / Value-Investing-Agent

核心思想: 像巴菲特/段永平/李录一样分析股票，关注:
  1. 护城河评分 (Moat Score)
  2. 内在价值估算 (Intrinsic Value via DCF/Graham公式)
  3. 安全边际 (Margin of Safety)
  4. 管理层质量 (基于财务指标推断)
  5. 行业地位 (基于市值/ROE在行业中的位置)

输出:
  - moat_score: 0~100
  - intrinsic_value: 元/股
  - margin_of_safety: 百分比
  - value_verdict: "低估" / "合理" / "高估"
  - investment_case: 一段总结性文字
"""

import math
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ValueAnalysisResult:
    moat_score: float                # 0~100
    moat_level: str                  # "宽" / "窄" / "无"
    intrinsic_value: float           # 元/股
    current_price: float             # 元/股
    margin_of_safety: float          # 百分比(正值=有安全边际)
    value_verdict: str               # "低估" / "合理" / "高估"
    investment_case: str             # 投资论点
    graham_number: float             # 格雷厄姆数
    dcf_value: float                 # DCF估值
    pe_value: float                  # PE估值(基于行业均值)
    pb_value: float                  # PB估值
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str


class ValueInvestingAnalyzer:
    """深度价值投资分析器"""

    # 行业平均PE/PB(简化假设)
    INDUSTRY_PE = 20.0
    INDUSTRY_PB = 3.0

    def analyze(self, quote: dict, financials: dict,
                valuation: dict, symbol: str) -> ValueAnalysisResult:
        """执行深度价值分析

        Args:
            quote: 实时行情
            financials: 财务数据
            valuation: 估值数据
            symbol: 股票代码

        Returns:
            ValueAnalysisResult
        """
        if not quote or 'error' in quote:
            return ValueAnalysisResult(
                moat_score=0, moat_level="无", intrinsic_value=0,
                current_price=0, margin_of_safety=0, value_verdict="数据不足",
                investment_case="无行情数据", graham_number=0, dcf_value=0,
                pe_value=0, pb_value=0, strengths=[], weaknesses=[],
                recommendation="无数据",
            )

        price = quote.get('price', 0)
        pe = quote.get('pe_ttm') or quote.get('pe', 0)
        pb = quote.get('pb', 0)

        # 1. 护城河评分
        moat_score, moat_level, strengths, weaknesses = self._moat_score(financials)

        # 2. 格雷厄姆数
        graham = self._graham_number(financials)

        # 3. DCF估值(简化)
        dcf = self._dcf_value(financials, quote)

        # 4. PE估值
        pe_val = self._pe_valuation(financials, quote)

        # 5. PB估值
        pb_val = self._pb_valuation(financials, quote)

        # 6. 综合内在价值(取多个方法的加权平均)
        values = [v for v in [graham, dcf, pe_val, pb_val] if v and v > 0]
        intrinsic = sum(values) / len(values) if values else 0

        # 7. 安全边际
        mos = (intrinsic - price) / price * 100 if price > 0 else 0

        # 8. 价值判定
        if mos >= 30:
            verdict = "低估"
        elif mos >= 10:
            verdict = "合理偏低"
        elif mos >= -10:
            verdict = "合理"
        elif mos >= -30:
            verdict = "合理偏高"
        else:
            verdict = "高估"

        # 9. 投资论点
        case = self._investment_case(
            symbol, price, intrinsic, mos, moat_score,
            moat_level, strengths, weaknesses, verdict,
        )

        return ValueAnalysisResult(
            moat_score=round(moat_score, 1),
            moat_level=moat_level,
            intrinsic_value=round(intrinsic, 2),
            current_price=price,
            margin_of_safety=round(mos, 1),
            value_verdict=verdict,
            investment_case=case,
            graham_number=round(graham, 2),
            dcf_value=round(dcf, 2),
            pe_value=round(pe_val, 2),
            pb_value=round(pb_val, 2),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=self._recommendation(verdict, mos, moat_score),
        )

    def _moat_score(self, financials: dict) -> tuple:
        """计算护城河评分

        参考巴菲特/晨星护城河框架:
          - 高ROE(资本回报率)
          - 高毛利率(定价权)
          - 稳定的ROE(一致性)
          - 低负债(财务稳健)
          - 高现金流/利润比(利润质量)
        """
        if not financials or 'error' in financials:
            return 0, "无", [], ["无财务数据"]

        score = 0
        strengths = []
        weaknesses = []

        # ROE (0~30)
        roe = financials.get('roe')
        if roe is not None:
            if roe >= 25:
                score += 30
                strengths.append(f"ROE={roe:.1f}%，极高资本回报，具备宽护城河特征")
            elif roe >= 20:
                score += 25
                strengths.append(f"ROE={roe:.1f}%，高资本回报")
            elif roe >= 15:
                score += 20
                strengths.append(f"ROE={roe:.1f}%，资本回报良好")
            elif roe >= 10:
                score += 12
            else:
                weaknesses.append(f"ROE={roe:.1f}%，资本回报不足")

        # 毛利率 (0~25)
        gm = financials.get('gross_margin')
        if gm is not None:
            if gm >= 60:
                score += 25
                strengths.append(f"毛利率={gm:.1f}%，极强定价权")
            elif gm >= 50:
                score += 20
                strengths.append(f"毛利率={gm:.1f}%，较强定价权")
            elif gm >= 40:
                score += 15
            elif gm >= 25:
                score += 8
            else:
                weaknesses.append(f"毛利率={gm:.1f}%，低毛利行业，无定价权")

        # 净利率 (0~15)
        nm = financials.get('net_margin')
        if nm is not None:
            if nm >= 30:
                score += 15
                strengths.append(f"净利率={nm:.1f}%，利润率高")
            elif nm >= 20:
                score += 12
            elif nm >= 10:
                score += 8
            else:
                weaknesses.append(f"净利率={nm:.1f}%，利润率偏低")

        # 现金流质量 (0~15)
        ocf = financials.get('ocf_per_share')
        eps = financials.get('eps')
        if ocf is not None and eps is not None and eps > 0:
            ratio = ocf / eps
            if ratio >= 1.0:
                score += 15
                strengths.append(f"现金流/利润={ratio:.2f}，利润全部转化为现金")
            elif ratio >= 0.8:
                score += 12
            elif ratio >= 0.5:
                score += 8
            else:
                weaknesses.append(f"现金流/利润={ratio:.2f}，利润质量差")

        # 成长性 (0~15)
        rev_yoy = financials.get('revenue_yoy')
        np_yoy = financials.get('net_profit_yoy')
        if rev_yoy is not None:
            if rev_yoy > 20:
                score += 5
            elif rev_yoy > 10:
                score += 3
        if np_yoy is not None:
            if np_yoy > 20:
                score += 5
            elif np_yoy > 10:
                score += 3

        score = min(score, 100)

        if score >= 60:
            level = "宽"
        elif score >= 35:
            level = "窄"
        else:
            level = "无"

        return score, level, strengths, weaknesses

    def _graham_number(self, financials: dict) -> float:
        """格雷厄姆数 = 22.5 * EPS + 2.5 * PS

        Graham建议: 合理价格 = 22.5倍EPS + 2.5倍每股净资产
        """
        if not financials or 'error' in financials:
            return 0

        eps = financials.get('eps')
        bps = financials.get('bps') or financials.get('total_equity')

        if bps and bps > 0 and eps is not None:
            return 22.5 * eps + 2.5 * bps
        elif eps:
            return 22.5 * eps
        return 0

    def _dcf_value(self, financials: dict, quote: dict) -> float:
        """简化DCF估值

        基于当前EPS，假设未来5年稳定增长，之后永续增长
        """
        if not financials or 'error' in financials:
            return 0

        eps = financials.get('eps')
        if not eps or eps <= 0:
            return 0

        # 增长率估计(基于历史增长)
        np_yoy = financials.get('net_profit_yoy')
        if np_yoy is not None:
            growth = max(min(np_yoy / 100, 0.2), -0.1)  # 限制在-10%~20%
        else:
            growth = 0.05  # 默认5%

        # 折现率
        discount_rate = 0.10

        # 永续增长率
        terminal_growth = 0.02

        # DCF = sum(eps*(1+g)^t / (1+r)^t, t=1..5) + terminal / (1+r)^5
        pv = 0
        for t in range(1, 6):
            pv += eps * (1 + growth) ** t / (1 + discount_rate) ** t

        # 终值
        terminal = eps * (1 + growth) ** 5 * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv += terminal / (1 + discount_rate) ** 5

        return pv

    def _pe_valuation(self, financials: dict, quote: dict) -> float:
        """基于行业平均PE的估值"""
        if not financials or 'error' in financials:
            return 0
        eps = financials.get('eps')
        if eps and eps > 0:
            return eps * self.INDUSTRY_PE
        return 0

    def _pb_valuation(self, financials: dict, quote: dict) -> float:
        """基于行业平均PB的估值"""
        if not financials or 'error' in financials:
            return 0
        bps = financials.get('bps') or financials.get('total_equity')
        if bps and bps > 0:
            return bps * self.INDUSTRY_PB
        return 0

    def _investment_case(self, symbol: str, price: float, intrinsic: float,
                         mos: float, moat_score: float, moat_level: str,
                         strengths: list, weaknesses: list,
                         verdict: str) -> str:
        """构建投资论点"""
        lines = []
        lines.append(f"【价值投资分析】")
        lines.append(f"内在价值={intrinsic:.2f}元 | 现价={price:.2f}元 | 安全边际={mos:+.1f}%")
        lines.append(f"护城河: {moat_level}(评分{moat_score:.0f}) | 价值判定: {verdict}")

        if strengths:
            lines.append(f"优势: {'; '.join(strengths[:3])}")
        if weaknesses:
            lines.append(f"风险: {'; '.join(weaknesses[:3])}")

        return "\n".join(lines)

    def _recommendation(self, verdict: str, mos: float, moat_score: float) -> str:
        if verdict == "低估" and moat_score >= 50:
            return "价值凸显，护城河清晰，建议逐步建仓"
        elif verdict == "低估":
            return "价格低于内在价值，但需评估护城河质量"
        elif verdict == "合理偏低" and moat_score >= 40:
            return "合理偏低，可小仓位试探"
        elif verdict == "合理":
            return "价格基本合理，持有观望"
        elif verdict == "高估":
            return "价格高于内在价值，建议减仓或止盈"
        else:
            return "结合趋势分析判断"