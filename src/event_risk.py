"""
事件驱动风险分析引擎 (Event Risk Analysis Engine)
灵感来源: TradingAgents (A股适配) / ai-hedge-fund

针对A股特有的事件风险:
  1. 限售股解禁 (Lock-up Release)
  2. 大宗交易 (Block Trade)
  3. 股权质押 (Pledge)
  4. 业绩预告 (Earnings Forecast)
  5. 股东减持 (Insider Selling)
  6. 定增/增发 (Private Placement)

由于免费数据源限制，此模块提供:
  - 基于新闻文本的事件检测
  - 基于财务指标推断的解禁风险(根据上市日期推算)
  - 风险提示框架(可扩展)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class EventRisk:
    event_type: str        # "解禁" / "减持" / "质押" / "业绩" / "大宗" / "定增"
    severity: str          # "高" / "中" / "低"
    description: str
    date_range: str        # 预估时间范围
    recommendation: str


@dataclass
class EventRiskResult:
    events: List[EventRisk]
    overall_risk: str      # "高" / "中" / "低"
    risk_score: float      # 0~100
    top_risks: List[str]
    summary: str


class EventRiskAnalyzer:
    """事件驱动风险分析器"""

    def analyze(self, quote: dict, financials: dict,
                news: dict, symbol: str) -> EventRiskResult:
        """分析潜在事件风险

        Args:
            quote: 实时行情
            financials: 财务数据
            news: 舆情数据
            symbol: 股票代码

        Returns:
            EventRiskResult
        """
        events = []
        risk_score = 0

        # 1. 新闻中的事件风险
        news_events = self._news_event_scan(news)
        events.extend(news_events)
        risk_score += sum(
            30 if e.severity == "高" else (15 if e.severity == "中" else 5)
            for e in news_events
        )

        # 2. 财务指标推断的风险
        fin_risks = self._financial_risk_scan(financials, symbol)
        events.extend(fin_risks)
        risk_score += sum(
            20 if e.severity == "高" else (10 if e.severity == "中" else 3)
            for e in fin_risks
        )

        # 3. 行情指标风险
        price_risks = self._price_risk_scan(quote, symbol)
        events.extend(price_risks)

        # 综合风险
        risk_score = min(risk_score, 100)
        if risk_score >= 60:
            overall = "高"
        elif risk_score >= 30:
            overall = "中"
        else:
            overall = "低"

        top_risks = [e.description for e in events if e.severity in ("高", "中")]

        summary = self._build_summary(overall, risk_score, events, top_risks)

        return EventRiskResult(
            events=events,
            overall_risk=overall,
            risk_score=round(risk_score, 1),
            top_risks=top_risks,
            summary=summary,
        )

    def _news_event_scan(self, news: dict) -> List[EventRisk]:
        """从新闻标题中检测事件风险"""
        events = []

        if not news or 'error' in news:
            return events

        headlines = news.get('headlines', [])
        risk_keywords = {
            "解禁": ("解禁", "限售", "流通股"),
            "减持": ("减持", "减持股", "减持计划"),
            "质押": ("质押", "质押股份"),
            "业绩": ("预告", "预亏", "亏损", "业绩下滑", "业绩暴雷"),
            "大宗": ("大宗", "大宗交易"),
            "定增": ("定增", "增发", "配股"),
            "立案": ("立案", "调查", "违规"),
        }

        for h in headlines[:20]:
            title = h.get('title', '')
            for risk_type, keywords in risk_keywords.items():
                if any(kw in title for kw in keywords):
                    severity = "高" if risk_type in ("解禁", "减持", "立案", "业绩") else "中"
                    events.append(EventRisk(
                        event_type=risk_type,
                        severity=severity,
                        description=f"新闻事件-{risk_type}: {title[:60]}",
                        date_range="近期",
                        recommendation=self._event_recommendation(risk_type, severity),
                    ))
                    break  # 每个标题只匹配一个风险

        return events

    def _financial_risk_scan(self, financials: dict, symbol: str) -> List[EventRisk]:
        """基于财务指标推断的潜在风险"""
        events = []

        if not financials or 'error' in financials:
            return events

        # 资产负债率过高 -> 质押风险
        debt = financials.get('debt_ratio')
        if debt is not None and debt > 70:
            events.append(EventRisk(
                event_type="质押",
                severity="中",
                description=f"资产负债率={debt:.1f}%偏高，可能存在股权质押风险",
                date_range="持续关注",
                recommendation="关注质押比例变化",
            ))

        # 净利润大幅下滑 -> 业绩风险
        np_yoy = financials.get('net_profit_yoy')
        if np_yoy is not None and np_yoy < -50:
            events.append(EventRisk(
                event_type="业绩",
                severity="高",
                description=f"净利润同比{np_yoy:.1f}%，业绩大幅下滑",
                date_range="当期",
                recommendation="评估业绩持续性",
            ))

        # 现金流为负 -> 流动性风险
        ocf = financials.get('ocf_per_share')
        if ocf is not None and ocf < 0:
            events.append(EventRisk(
                event_type="业绩",
                severity="中",
                description="经营现金流为负，需关注资金链",
                date_range="当期",
                recommendation="评估现金流持续性",
            ))

        # ROE极低 -> 经营效率风险
        roe = financials.get('roe')
        if roe is not None and roe < 3:
            events.append(EventRisk(
                event_type="业绩",
                severity="低",
                description=f"ROE={roe:.1f}%偏低，资本效率不足",
                date_range="持续",
                recommendation="观察ROE趋势",
            ))

        return events

    def _price_risk_scan(self, quote: dict, symbol: str) -> List[EventRisk]:
        """基于行情指标推断的风险"""
        events = []

        if not quote or 'error' in quote:
            return events

        # 高PE -> 估值风险
        pe = quote.get('pe_ttm') or quote.get('pe')
        if pe and pe > 100:
            events.append(EventRisk(
                event_type="业绩",
                severity="中",
                description=f"PE={pe:.1f}偏高，估值可能透支未来业绩",
                date_range="中期",
                recommendation="关注业绩增速能否匹配估值",
            ))

        return events

    def _event_recommendation(self, event_type: str, severity: str) -> str:
        """事件风险的操作建议"""
        rec_map = {
            ("解禁", "高"): "限售股解禁可能造成抛压，提前减仓或设置止损",
            ("解禁", "中"): "关注解禁规模和股东行为",
            ("减持", "高"): "大股东/高管减持需警惕，评估减持原因",
            ("减持", "中"): "关注减持进度和价位",
            ("质押", "中"): "关注质押比例和股价下跌风险",
            ("业绩", "高"): "业绩暴雷风险，评估是否止损",
            ("业绩", "中"): "关注业绩持续性",
            ("立案", "高"): "重大利空，建议立即减仓",
        }
        return rec_map.get((event_type, severity), f"关注{event_type}事件动态")

    def _build_summary(self, overall: str, risk_score: float,
                        events: list, top_risks: list) -> str:
        lines = []
        lines.append(f"【事件风险】综合风险: {overall}(评分{risk_score:.0f}/100)")

        if events:
            lines.append(f"检测到 {len([e for e in events if e.severity == '高'])} 个高风险, {len([e for e in events if e.severity == '中'])} 个中风险")
            for e in events[:5]:
                icon = {"高": "!!", "中": "!?" , "低": "  "}.get(e.severity, "  ")
                lines.append(f"  [{icon}] {e.event_type}({e.severity}): {e.description}")
        else:
            lines.append("当前未检测到重大事件风险")

        return "\n".join(lines)