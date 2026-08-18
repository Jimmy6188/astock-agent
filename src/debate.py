"""
多空对抗辩论引擎 (Bull-Bear Debate Engine)
灵感来源: TradingAgents / ai-berkshire / ai-hedge-fund

核心思想: 让"多头"和"空头"从多个维度各自主张观点，
通过结构化的正反方论点，帮助用户看到完整的市场画面。

辩论维度:
  1. 技术面 (多头 vs 空头)
  2. 基本面 (多头 vs 空头)
  3. 资金面 (多头 vs 空头)
  4. 舆情面 (多头 vs 空头)

每个维度生成2-3条论点，附证据。
最终汇总"多头总分" vs "空头总分"，给出倾向性结论。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class DebateArgument:
    side: str          # "bull" or "bear"
    dimension: str
    argument: str
    evidence: str = ""


@dataclass
class DebateResult:
    bull_score: int
    bear_score: int
    bull_arguments: List[DebateArgument]
    bear_arguments: List[DebateArgument]
    verdict: str       # "多头占优" / "空头占优" / "势均力敌"
    confidence: str
    bull_thesis: str   # 多头核心论点
    bear_thesis: str   # 空头核心论点
    summary: str


class DebateEngine:
    """多空对抗辩论引擎"""

    def debate(self, quote: dict, kline: pd.DataFrame,
               financials: dict, valuation: dict,
               money_flow: dict, news: dict, lhb: list = None) -> DebateResult:
        """执行多空辩论

        Args:
            quote: 实时行情
            kline: 历史K线
            financials: 财务数据
            valuation: 估值数据
            money_flow: 资金流数据
            news: 舆情数据
            lhb: 龙虎榜数据

        Returns:
            DebateResult
        """
        if lhb is None:
            lhb = []

        bull_args = []
        bear_args = []
        bull_points = 0
        bear_points = 0

        # ---- 1. 技术面辩论 ----
        tech_bull, tech_bear, b_pts, be_pts = self._tech_debate(kline, quote)
        bull_args.extend(tech_bull)
        bear_args.extend(tech_bear)
        bull_points += b_pts
        bear_points += be_pts

        # ---- 2. 基本面辩论 ----
        fund_bull, fund_bear, b_pts, be_pts = self._fundamental_debate(financials, quote)
        bull_args.extend(fund_bull)
        bear_args.extend(fund_bear)
        bull_points += b_pts
        bear_points += be_pts

        # ---- 3. 资金面辩论 ----
        mf_bull, mf_bear, b_pts, be_pts = self._moneyflow_debate(money_flow, quote, lhb)
        bull_args.extend(mf_bull)
        bear_args.extend(mf_bear)
        bull_points += b_pts
        bear_points += be_pts

        # ---- 4. 舆情面辩论 ----
        sent_bull, sent_bear, b_pts, be_pts = self._sentiment_debate(news, quote)
        bull_args.extend(sent_bull)
        bear_args.extend(sent_bear)
        bull_points += b_pts
        bear_points += be_pts

        # ---- 判定 ----
        total = bull_points + bear_points
        if total == 0:
            verdict = "势均力敌"
            confidence = "低"
        elif bull_points > bear_points:
            diff = bull_points - bear_points
            if diff >= 3:
                verdict = "多头占优"
                confidence = "高"
            elif diff >= 2:
                verdict = "多头略优"
                confidence = "中"
            else:
                verdict = "多头微优"
                confidence = "低"
        else:
            diff = bear_points - bull_points
            if diff >= 3:
                verdict = "空头占优"
                confidence = "高"
            elif diff >= 2:
                verdict = "空头略优"
                confidence = "中"
            else:
                verdict = "空头微优"
                confidence = "低"

        # 核心论点
        bull_thesis = bull_args[0].argument if bull_args else "缺乏多头证据"
        bear_thesis = bear_args[0].argument if bear_args else "缺乏空头证据"

        summary = self._build_summary(verdict, confidence, bull_points, bear_points,
                                       bull_args, bear_args)

        return DebateResult(
            bull_score=bull_points,
            bear_score=bear_points,
            bull_arguments=bull_args,
            bear_arguments=bear_args,
            verdict=verdict,
            confidence=confidence,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            summary=summary,
        )

    def _tech_debate(self, kline: pd.DataFrame, quote: dict) -> Tuple:
        """技术面辩论"""
        bull = []
        bear = []
        b_pts = 0
        be_pts = 0

        if kline is None or kline.empty or len(kline) < 20:
            return bull, bear, b_pts, be_pts

        closes = kline['close'].values if 'close' in kline.columns else None
        volumes = kline['volume'].values if 'volume' in kline.columns else None

        if closes is not None:
            # 短期均线趋势
            ma5 = closes[-5:].mean() if len(closes) >= 5 else None
            ma10 = closes[-10:].mean() if len(closes) >= 10 else None
            ma20 = closes[-20:].mean() if len(closes) >= 20 else None

            if ma5 and ma10 and ma20:
                if ma5 > ma10 > ma20:
                    bull.append(DebateArgument("bull", "技术面",
                        "短期均线多头排列(MA5>MA10>MA20)，上升趋势确认",
                        f"MA5={ma5:.2f} > MA10={ma10:.2f} > MA20={ma20:.2f}"))
                    b_pts += 2
                elif ma5 < ma10 < ma20:
                    bear.append(DebateArgument("bear", "技术面",
                        "短期均线空头排列(MA5<MA10<MA20)，下降趋势确认",
                        f"MA5={ma5:.2f} < MA10={ma10:.2f} < MA20={ma20:.2f}"))
                    be_pts += 2
                else:
                    bull.append(DebateArgument("bull", "技术面",
                        "均线交错，短期趋势不明朗但未见明显下跌",
                        f"MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}"))
                    b_pts += 0.5

            # 成交量
            if volumes is not None:
                avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
                current_vol = volumes[-1] if len(volumes) > 0 else 0
                if current_vol > avg_vol * 1.5:
                    bull.append(DebateArgument("bull", "技术面",
                        "当前成交量明显放大(>1.5倍均量)，资金关注度高",
                        f"当前量={current_vol:.0f}, 20日均量={avg_vol:.0f}"))
                    b_pts += 1
                elif current_vol < avg_vol * 0.5:
                    bear.append(DebateArgument("bear", "技术面",
                        "成交量明显萎缩(<0.5倍均量)，市场关注度不足",
                        f"当前量={current_vol:.0f}, 20日均量={avg_vol:.0f}"))
                    be_pts += 1

            # 价格位置
            if len(closes) >= 20:
                low_20 = np.min(closes[-20:])
                high_20 = np.max(closes[-20:])
                cur = closes[-1]
                pos = (cur - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
                if pos > 0.7:
                    bear.append(DebateArgument("bear", "技术面",
                        f"股价位于近20日高位区({pos*100:.0f}%)，短期上行空间有限",
                        f"20日内{pos*100:.0f}%位置"))
                    be_pts += 1
                elif pos < 0.3:
                    bull.append(DebateArgument("bull", "技术面",
                        f"股价位于近20日低位区({pos*100:.0f}%)，接近支撑位，反弹概率较高",
                        f"20日内{pos*100:.0f}%位置"))
                    b_pts += 1

        return bull, bear, b_pts, be_pts

    def _fundamental_debate(self, financials: dict, quote: dict) -> Tuple:
        """基本面辩论"""
        bull = []
        bear = []
        b_pts = 0
        be_pts = 0

        if not financials or 'error' in financials:
            return bull, bear, b_pts, be_pts

        # ROE
        roe = financials.get('roe')
        if roe is not None:
            if roe >= 15:
                bull.append(DebateArgument("bull", "基本面",
                    f"ROE={roe:.1f}%，盈利能力优秀，符合巴菲特标准(>15%)",
                    f"ROE={roe:.1f}%"))
                b_pts += 2
            elif roe >= 10:
                bull.append(DebateArgument("bull", "基本面",
                    f"ROE={roe:.1f}%，盈利能力良好",
                    f"ROE={roe:.1f}%"))
                b_pts += 1
            elif roe < 5:
                bear.append(DebateArgument("bear", "基本面",
                    f"ROE={roe:.1f}%，盈利能力偏弱，资本回报不足",
                    f"ROE={roe:.1f}%"))
                be_pts += 1

        # 毛利率
        gm = financials.get('gross_margin')
        if gm is not None:
            if gm >= 50:
                bull.append(DebateArgument("bull", "基本面",
                    f"毛利率={gm:.1f}%，高毛利壁垒，具备护城河特征",
                    f"毛利率={gm:.1f}%"))
                b_pts += 1
            elif gm < 20:
                bear.append(DebateArgument("bear", "基本面",
                    f"毛利率={gm:.1f}%，低毛利行业，利润空间有限",
                    f"毛利率={gm:.1f}%"))
                be_pts += 1

        # 营收增长
        rev_yoy = financials.get('revenue_yoy')
        if rev_yoy is not None:
            if rev_yoy > 20:
                bull.append(DebateArgument("bull", "基本面",
                    f"营收同比+{rev_yoy:.1f}%，高速增长",
                    f"营收YoY=+{rev_yoy:.1f}%"))
                b_pts += 2
            elif rev_yoy > 5:
                bull.append(DebateArgument("bull", "基本面",
                    f"营收同比+{rev_yoy:.1f}%，稳健增长",
                    f"营收YoY=+{rev_yoy:.1f}%"))
                b_pts += 1
            elif rev_yoy < -10:
                bear.append(DebateArgument("bear", "基本面",
                    f"营收同比{rev_yoy:.1f}%，业务萎缩",
                    f"营收YoY={rev_yoy:.1f}%"))
                be_pts += 2

        # 净利润增长
        np_yoy = financials.get('net_profit_yoy')
        if np_yoy is not None:
            if np_yoy > 30:
                bull.append(DebateArgument("bull", "基本面",
                    f"净利润同比+{np_yoy:.1f}%，业绩爆发",
                    f"净利YoY=+{np_yoy:.1f}%"))
                b_pts += 2
            elif np_yoy < -20:
                bear.append(DebateArgument("bear", "基本面",
                    f"净利润同比{np_yoy:.1f}%，业绩大幅下滑",
                    f"净利YoY={np_yoy:.1f}%"))
                be_pts += 2

        # 负债率
        debt = financials.get('debt_ratio')
        if debt is not None:
            if debt > 70:
                bear.append(DebateArgument("bear", "基本面",
                    f"资产负债率={debt:.1f}%，负债率偏高，财务风险较大",
                    f"负债率={debt:.1f}%"))
                be_pts += 1
            elif debt < 30:
                bull.append(DebateArgument("bull", "基本面",
                    f"资产负债率={debt:.1f}%，财务结构健康",
                    f"负债率={debt:.1f}%"))
                b_pts += 0.5

        # 现金流
        ocf = financials.get('ocf_per_share')
        eps = financials.get('eps')
        if ocf is not None and eps is not None and eps != 0:
            ocf_ratio = ocf / eps
            if ocf_ratio > 0.8:
                bull.append(DebateArgument("bull", "基本面",
                    "经营现金流/每股收益>0.8，利润质量高(真金白银)",
                    f"OCF/EPS={ocf_ratio:.2f}"))
                b_pts += 1
            elif ocf_ratio < 0:
                bear.append(DebateArgument("bear", "基本面",
                    "经营现金流为负，利润未转化为现金",
                    f"OCF/EPS={ocf_ratio:.2f}"))
                be_pts += 1

        return bull, bear, b_pts, be_pts

    def _moneyflow_debate(self, money_flow: dict, quote: dict, lhb: list) -> Tuple:
        """资金面辩论"""
        bull = []
        bear = []
        b_pts = 0
        be_pts = 0

        if money_flow and 'error' not in money_flow:
            main_net = money_flow.get('main_net_inflow', 0)
            main_pct = money_flow.get('main_net_pct', 0)

            if main_net > 0:
                bull.append(DebateArgument("bull", "资金面",
                    f"主力净流入{main_net/1e4:.0f}万元(占比{main_pct:.1f}%)，资金做多意愿强",
                    f"主力净流入={main_net/1e4:.0f}万"))
                b_pts += 2
            elif main_net < 0:
                bear.append(DebateArgument("bear", "资金面",
                    f"主力净流出{abs(main_net)/1e4:.0f}万元(占比{abs(main_pct):.1f}%)，资金在撤离",
                    f"主力净流出={abs(main_net)/1e4:.0f}万"))
                be_pts += 2
        else:
            bull.append(DebateArgument("bull", "资金面",
                "资金流数据暂不可用(东方财富限流)，以下判断仅供参考", ""))

        # 龙虎榜
        if lhb and len(lhb) > 0:
            buy_total = sum(r.get('buy_amount', 0) for r in lhb)
            sell_total = sum(r.get('sell_amount', 0) for r in lhb)
            net = buy_total - sell_total
            if net > 0:
                bull.append(DebateArgument("bull", "资金面",
                    f"近两月龙虎榜净买入{net/1e4:.0f}万元，机构/游资看好",
                    f"龙虎榜净买入={net/1e4:.0f}万"))
                b_pts += 1
            else:
                bear.append(DebateArgument("bear", "资金面",
                    f"近两月龙虎榜净卖出{abs(net)/1e4:.0f}万元，短线资金减仓",
                    f"龙虎榜净卖出={abs(net)/1e4:.0f}万"))
                be_pts += 1

        return bull, bear, b_pts, be_pts

    def _sentiment_debate(self, news: dict, quote: dict) -> Tuple:
        """舆情面辩论"""
        bull = []
        bear = []
        b_pts = 0
        be_pts = 0

        if news and 'error' not in news and news.get('news_count', 0) > 0:
            sentiment = news.get('sentiment', 'neutral')
            headlines = news.get('headlines', [])

            if sentiment == 'positive':
                bull.append(DebateArgument("bull", "舆情面",
                    "近期新闻整体偏正面，市场情绪利好",
                    f"情绪={sentiment}, 新闻数={news.get('news_count',0)}"))
                b_pts += 2
            elif news.get('sentiment') == 'negative':
                bear.append(DebateArgument("bear", "舆情面",
                    "近期新闻整体偏负面，市场情绪承压",
                    f"情绪=negative"))
                be_pts += 2

            # 标题分析
            for h in news.get('headlines', [])[:5]:
                title = h.get('title', '')
                if any(w in title for w in ['大涨', '突破', '利好', '创新高', '业绩超预期', '收购']):
                    bull.append(DebateArgument("bull", "舆情面",
                        f"正面新闻: {title[:50]}", title))
                    b_pts += 0.5
                elif any(w in title for w in ['大跌', '暴跌', '利空', '减持', '亏损', '下调']):
                    bear.append(DebateArgument("bear", "舆情面",
                        f"负面新闻: {title[:50]}", title))
                    be_pts += 0.5
        else:
            bull.append(DebateArgument("bull", "舆情面",
                "舆情数据暂不可用，以下判断仅供参考", ""))

        return bull, bear, b_pts, be_pts

    def _build_summary(self, verdict: str, confidence: str,
                        bull_pts: int, bear_pts: int,
                        bull_args: list, bear_args: list) -> str:
        """构建辩论总结"""
        lines = []
        lines.append(f"【多空辩论】{verdict}(置信度:{confidence}) | 多头{bull_pts}分 vs 空头{bear_pts}分")
        lines.append("")

        if bull_args:
            lines.append("=== 多头观点 ===")
            for a in bull_args[:4]:
                lines.append(f"  [{a.dimension}] {a.argument}")
            lines.append("")

        if bear_args:
            lines.append("=== 空头观点 ===")
            for a in bear_args[:4]:
                lines.append(f"  [{a.dimension}] {a.argument}")

        return "\n".join(lines)