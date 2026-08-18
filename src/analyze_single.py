"""单只股票分析脚本 - 强制代码校验"""
import sys, io, json, time, requests, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_provider import AStockDataProvider, verify_symbol
from analyzer import StockAnalyzer

# ====== 强制代码校验 ======
# 必须使用真实代码,不得杜撰。如果不确定代码,必须先联网搜索确认。
symbol = "600460"   # 股票代码 (6位)
name = "士兰微"     # 股票名称 (用于交叉验证)

# 先验证代码
print(f"验证代码: {symbol} (预期名称: {name})...")
verify = verify_symbol(symbol, expected_name=name)
if not verify.get("valid"):
    print(f"\n{'='*60}")
    print(f"❌ 代码验证失败: {verify.get('error')}")
    print(f"{'='*60}")
    print(f"可能原因:")
    print(f"  1. 股票代码错误 - 请确认正确代码")
    print(f"  2. 股票名称与代码不匹配 - 请检查")
    print(f"  3. 网络问题 - 请重试")
    print(f"  4. 该股票已退市")
    sys.exit(1)

print(f"✅ 验证通过: {verify['name']} 现价 {verify['price']}")
print(f"\n{'='*60}")
print(f"  {symbol} {verify['name']} 深度分析报告")
print(f"{'='*60}\n")

t0 = time.time()

# 分析
p = AStockDataProvider()
a = StockAnalyzer(p)
report = a.analyze(symbol)
quote = p.get_realtime_quote(symbol)
elapsed = time.time() - t0

print(f"用时: {elapsed:.1f}s\n")

# === 实时行情 ===
print(f"【实时行情】")
print(f"  现价: {quote.get('price',0)}  涨跌: {quote.get('change_pct',0)}%  换手: {quote.get('turnover_rate',0)}%")
print(f"  PE-TTM: {quote.get('pe_ttm','?')}  PB: {quote.get('pb','?')}\n")

# === 综合评分 ===
print(f"【综合评分】{report.get('overall_score')}分 / {report.get('overall_rating')}")
print(f"  操作建议: {report.get('action_suggestion','')}\n")

# === 六维分析 ===
print(f"【六维分析】")
for d in report.get('dimensions', []):
    dim = d.get('dimension') if isinstance(d, dict) else d.name
    score = d.get('score') if isinstance(d, dict) else d.score
    rating = d.get('rating') if isinstance(d, dict) else d.rating
    print(f"  {dim:<10} {score:>5.1f}分 ({rating})")
    for s in d.get('signals', [])[:2]:
        print(f"    → {s}")
print()

# === 关键信号 ===
print(f"【关键信号】")
for i, s in enumerate(report.get('key_signals', []), 1):
    print(f"  {i}. {s}")
print()

# === 通达信-提前抄底 ===
tdx = report.get('tdx_signals') or {}
if tdx:
    print(f"【通达信-提前抄底】")
    print(f"  综合: {tdx.get('综合结论','?')}({tdx.get('综合得分',0)}分)")
    print(f"  多方力度: {tdx.get('多方力度',0)} ({tdx.get('多方力度趋势','')})")
    print(f"  空方力度: {tdx.get('空方力度',0)} ({tdx.get('空方力度趋势','')})")
    print(f"  持股天数: {tdx.get('持股天数',0)}日")
    for b in tdx.get('买入信号', []): print(f"  🟢 {b}")
    for s in tdx.get('卖出信号', []): print(f"  🔴 {s}")
    for n in tdx.get('中性信号', []): print(f"  ⚪ {n}")
    print(f"  建议: {tdx.get('建议','')}\n")

# === 通达信综合指标 ===
tdxi = report.get('tdx_indicators') or {}
if tdxi:
    print(f"【通达信综合指标】{tdxi.get('combined_verdict','?')} (置信度:{tdxi.get('confidence','?')})")
    for s in tdxi.get('signals', [])[:5]:
        print(f"  → {s}")
    print()

# === 多空辩论 ===
debate = report.get('debate') or {}
if debate:
    print(f"【多空辩论】{debate.get('verdict','?')} (多{debate.get('bull_score',0)} vs 空{debate.get('bear_score',0)})")
    print(f"  {debate.get('summary','')[:200]}\n")

# === 价值投资 ===
vi = report.get('value_investing') or {}
if vi:
    print(f"【价值投资】估值: {vi.get('value_verdict','?')}")
    print(f"  内在价值: {vi.get('intrinsic_value','?')}  安全边际: {vi.get('margin_of_safety','?')}%")
    print(f"  护城河: {vi.get('moat_level','?')}")
    print(f"  投资案例: {vi.get('investment_case','')[:200]}\n")

# === 缠论 ===
chan = report.get('chan') or {}
if chan:
    buy = chan.get('buy_point') or ''
    sell = chan.get('sell_point') or ''
    buy_p = chan.get('buy_price')
    sell_p = chan.get('sell_price')
    buy_str = f"{buy} (参考价{buy_p}元)" if buy_p else buy
    sell_str = f"{sell} (参考价{sell_p}元)" if sell_p else sell
    print(f"【缠中说禅】位置:{chan.get('position','?')}  买入:{buy_str}  卖出:{sell_str}")
    print(f"  结构清晰度:{chan.get('structure_score',0)}分")
    print(f"  建议:{chan.get('recommendation','')}\n")

# === 事件风险 ===
er = report.get('event_risk') or {}
if er:
    print(f"【事件风险】{er.get('overall_risk','?')}")
    print(f"  {er.get('summary','')[:150]}\n")

# === 风险提示 ===
for w in report.get('risk_warnings', []):
    print(f"  ⚠️  {w}")

print(f"{'='*60}")