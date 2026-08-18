"""
持仓复盘脚本 - 对 portfolio.json 中所有非ETF个股进行深度分析
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from .data_provider import get_provider
from .analyzer import StockAnalyzer

p = get_provider()
a = StockAnalyzer(p)

# 全部13只持仓 (from portfolio.json)
holdings = [
    ('002580', '圣阳股份', 400, 10.71),
    ('562500', '机器人ETF华夏', 6500, 0.90),
    ('601016', '节能风电', 1000, 4.35),
    ('002160', '常铝股份', 1800, 5.70),
    ('002145', '钛能化学', 2245, 5.88),
    ('600016', '民生银行', 4555, -2.082),
    ('600785', '新华百货', 140, -22.094),
    ('002165', '红宝丽', 400, 4.19),
    ('588800', '科创100ETF华夏', 1000, 0.83),
    ('159162', '工业有色ETF鹏华', 3000, 0.92),
    ('002498', '汉缆股份', 1000, 8.46),
    ('600556', '天下秀', 2000, 7.23),
    ('600159', '大龙地产', 800, 9.46),
]

# Step 1: Get all quotes
print("=== Step 1: 获取行情 ===")
all_quotes = {}
for sym, name, qty, cost in holdings:
    q = p.get_realtime_quote(sym)
    all_quotes[sym] = q
    if isinstance(q, dict) and 'error' not in q:
        price = q.get('price', 0)
        pnl = (price / cost - 1) * 100 if cost > 0 else float('inf')
        chg = q.get('change_pct', 0)
        print(f"  {sym} {name}: {price} ({chg:+.1f}%) pnl={pnl:.1f}%")
    else:
        print(f"  {sym} {name}: ERROR - {q}")

# Step 2: Analyze non-ETF stocks
# ETF: 5xxxxx(上交所ETF), 159xxx(深交所ETF), 51xxxx(上交所ETF)
def _is_etf(sym: str) -> bool:
    s = str(sym).zfill(6)
    return s.startswith("5") or s.startswith("15")

non_etf = [h for h in holdings if not _is_etf(h[0])]
print(f"\n=== Step 2: 分析 {len(non_etf)} 只个股 ===")

results = []
for sym, name, qty, cost in non_etf:
    t0 = time.time()
    try:
        report = a.analyze(sym)
        elapsed = time.time() - t0
        quote = all_quotes.get(sym, {})
        price = quote.get('price', 0)
        pnl_pct = (price / cost - 1) * 100 if cost > 0 else float('inf')
        chg = quote.get('change_pct', 0)

        sc = report.get('tdx_signals', {}) or {}
        tdx = report.get('tdx_indicators', {}) or {}
        debate = report.get('debate', {}) or {}
        vi = report.get('value_investing', {}) or {}
        resonance = report.get('resonance', {}) or {}
        chan = report.get('chan', {}) or {}

        r = {
            'symbol': sym,
            'name': name,
            'quantity': qty,
            'cost_price': cost,
            'price': price,
            'change_pct': chg,
            'pnl_pct': round(pnl_pct, 1) if pnl_pct != float('inf') else None,
            'market_value': round(price * qty, 2),
            'score': report.get('overall_score'),
            'rating': report.get('overall_rating'),
            'suggestion': (report.get('action_suggestion') or '')[:120],
            'key_signals': report.get('key_signals', [])[:5],
            'warnings': report.get('risk_warnings', []),
            'tdx_signals': {
                'verdict': sc.get('综合结论', ''),
                'score': sc.get('综合得分', 0),
                'buy': sc.get('买入信号', []),
                'sell': sc.get('卖出信号', []),
                'neutral': sc.get('中性信号', []),
                'recommendation': sc.get('建议', ''),
            },
            'tdx_verdict': tdx.get('combined_verdict', ''),
            'debate': debate.get('verdict', ''),
            'resonance': resonance.get('direction', ''),
            'vi': vi.get('value_verdict', ''),
            'vi_intrinsic': vi.get('intrinsic_value'),
            'chan': chan.get('position', ''),
            'elapsed': round(elapsed, 1),
        }
        results.append(r)
        print(f"  {sym} {name}: {report.get('overall_rating')}({report.get('overall_score')}分) | "
              f"信号:{sc.get('综合结论','')} | 用时{elapsed:.1f}s")
    except Exception as e:
        print(f"  {sym} {name}: ERROR - {e}")
        results.append({'symbol': sym, 'name': name, 'error': str(e)})

# Step 3: ETF quotes only
print(f"\n=== Step 3: ETF行情 ===")
etf_quotes = []
for sym, name, qty, cost in holdings:
    if _is_etf(sym):
        q = all_quotes.get(sym, {})
        if isinstance(q, dict) and 'error' not in q:
            price = q.get('price', 0)
            pnl = (price / cost - 1) * 100
            chg = q.get('change_pct', 0)
            etf_quotes.append({
                'symbol': sym, 'name': name, 'quantity': qty, 'cost_price': cost,
                'price': price, 'change_pct': chg, 'pnl_pct': round(pnl, 1),
                'market_value': round(price * qty, 2),
            })
            print(f"  {sym} {name}: {price} ({chg:+.1f}%) pnl={pnl:.1f}%")
        elif 'error' not in q:
            print(f"  {sym} {name}: ERROR - {q}")

# Step 4: Save full results
output = {
    'review_date': time.strftime('%Y-%m-%d'),
    'review_time': time.strftime('%H:%M:%S'),
    'total_holdings': len(holdings),
    'non_etf_analyzed': len(results),
    'holdings_results': results,
    'etf_quotes': etf_quotes,
}

with open(r'C:/Users/sweet/.proma/agent-workspaces/astock-agent/portfolio_review.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"\n=== Done ===")
print(f"Results saved to portfolio_review.json")

# Print summary for display
print(f"\n{'='*60}")
print(f"复盘汇总 | {output['review_date']} {output['review_time']}")
print(f"{'='*60}")
print(f"{'股票':<8} {'名称':<8} {'现价':>6} {'涨跌幅':>6} {'盈亏%':>6} {'评分':>4} {'评级':<8} {'信号结论':<8}")
print(f"{'-'*60}")
for r in results:
    if 'error' in r:
        print(f"{r['symbol']:<8} {r.get('name',''):<8} {'ERROR'}")
        continue
    sc = r.get('tdx_signals', {})
    print(f"{r['symbol']:<8} {r.get('name',''):<8} {r.get('price',0):>6.2f} {r.get('change_pct',0):>+6.1f}% "
          f"{str(r.get('pnl_pct','')):>6} {str(r.get('score','')):>4} {str(r.get('rating','')):<8} {sc.get('verdict',''):<8}")

for e in etf_quotes:
    print(f"{e['symbol']:<8} {e.get('name',''):<8} {e.get('price',0):>6.4f} {e.get('change_pct',0):>+6.1f}% "
          f"{str(e.get('pnl_pct','')):>6}")