# 📋 astock-agent 项目代码质量审查报告

> **审查日期**: 2026-08-17  
> **项目版本**: 1.0.0  
> **审查范围**: 全部核心源码（`src/` 目录下 15 个 Python 文件，约 5000+ 行）

---

## 一、总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| **功能完整性** | ⭐⭐⭐⭐ | 六维分析、策略回测、MCP/CLI 双模式功能齐全 |
| **代码规范** | ⭐⭐⭐ | 基本可读，但存在多处不一致和冗余 |
| **健壮性** | ⭐⭐⭐ | 有异常处理但过于宽泛，部分边界情况未覆盖 |
| **安全性** | ⭐⭐ | 存在若干安全隐患需关注 |
| **性能** | ⭐⭐⭐ | 同步阻塞、N+1 查询问题明显 |
| **可维护性** | ⭐⭐⭐ | 模块划分清晰，但部分函数过长、耦合度偏高 |

**综合评级: B+（良好，有改进空间）**

---

## 二、🔴 严重问题（需尽快修复）

### 1. 死代码 — `data_provider.py` 第 102-113 行

```python
# _parse_tencent_row 函数中：
        return None   # ← 第100行已经return了

        # 以下代码永远不会执行！
        today_high = max(_sf(5), _sf(3), _sf(8))  # 死代码
        today_low = min(_sf(5), _sf(3), _sf(16))   # 死代码
        result["high"] = today_high                # 死代码
        result["low"] = today_low                  # 死代码
        return result                              # 死代码
    except Exception as e:                         # 这个except也是死的
```

**影响**: 虽然不影响运行，但说明代码有复制粘贴残留或重构不完整。

---

### 2. 重复导入 — `backtest.py` 第 10 行

```python
from dataclasses import dataclass, field   # 第9行
from dataclasses import dataclass, field   # 第10行（重复！）
```

---

### 3. 选股功能的 N+1 性能灾难 — `data_provider.py` 第 843-847 行

```python
valid_symbols = []
for _, r in df.iterrows():          # 遍历每只股票
    sym = str(r["代码"])
    if not _is_new_stock(sym):       # ← 每只股票都单独调用API查询历史K线！
        valid_symbols.append(sym)
```

**问题**: 如果筛选后有 100 只股票，就会发起 **100 次** Baostock/AKShare API 请求来判断是否为新股。全市场约 5000 只股票时更恐怖。

**建议**: 批量预加载或使用上市日期字段直接过滤。

---

### 4. 全局可变状态 + 线程安全风险

```python
# data_provider.py
_provider_instance = None      # 全局单例
_BS = None                     # Baostock 连接
_BS_LOGIN = False              # 登录状态
_meta_recorder = None          # MCP元数据回调

# mcp_server.py
_META = {}                     # 全局元数据字典
```

**问题**: 
- MCP Server 是异步框架，这些全局状态在并发请求下可能产生竞态条件
- `Baostock` 的登录状态没有锁保护

---

## 三、🟡 中等问题（建议修复）

### 5. 异常处理过于宽泛

项目中大量使用 `except Exception as e` 吞掉异常：

```python
# analyzer.py 第366-368行
except Exception as e:
    logger.error(f"技术面分析异常: {e}")
    details["error"] = str(e)   # 错误信息暴露给用户
```

**问题**:
- 可能隐藏真正的 bug（如 `KeyError`、`TypeError`）
- 异常信息直接返回给用户可能泄露内部实现细节

**建议**: 至少区分业务异常和程序错误。

---

### 6. 非相对导入 — `analyzer.py` 第 10 行

```python
from data_provider import AStockDataProvider   # 应该用相对导入
```

**问题**: 当模块被其他项目引用或测试时会报 `ModuleNotFoundError`。

**正确写法**: `from .data_provider import AStockDataProvider`

---

### 7. 缓存无上限 — 内存泄漏风险

```python
def __init__(self, cache_ttl: int = 300):
    self._cache: Dict[str, dict] = {}   # 无限增长！

def _set_cache(self, key: str, data):
    self._cache[key] = {"data": data, "ts": time.time()}  # 只进不出
```

**问题**: 长时间运行后缓存会无限膨胀，尤其 MCP Server 场景。

**建议**: 使用 LRU 策略或设置最大条目数。

---

### 8. 魔法数字泛滥

```python
# analyzer.py - 大量硬编码阈值
if pe < 15: score += 15           # 为什么是15？
elif pe < 25: score += 8          # 为什么是25？
if roe > 20: score += 15          # 为什么是20%？
if turnover > 10: score += 3      # 为什么是10%？

# backtest.py
commission_rate = 0.0003   # 万三
slippage = 0.001           # 千一
stamp_tax = 0.001          # 千一
```

**建议**: 提取为命名常量或配置项，并添加注释说明来源。

---

### 9. 函数过长 — 违反单一职责原则

| 文件 | 函数 | 行数 |
|------|------|------|
| `analyzer.py` | `TechnicalAnalyzer.analyze()` | ~320 行 |
| `analyzer.py` | `StockAnalyzer.analyze()` | ~230 行 |
| `data_provider.py` | `screen_stocks()` | ~110 行 |

**建议**: 拆分为多个子方法。

---

## 四、🔵 安全问题

### 10. User-Agent 过于简单易被封禁

```python
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
```

**风险**: 腾讯财经 API 可能识别并封禁爬虫请求。

---

### 11. 无 API 速率限制

所有外部 API 调用（腾讯、AKShare、Baostock）都没有限流机制。高频调用可能导致 IP 被封。

---

### 12. 敏感信息潜在泄露

```python
# pyproject.toml
authors = [
    {name = "astock-agent", email = "astock@example.com"}  # 占位邮箱
]
```
虽然当前是示例值，但生产环境需注意。

---

## 五、🟢 代码风格与最佳实践

### 13. 字符串引号混用

项目中同时使用单引号和双引号，且同一文件内也不统一：
```python
symbol = "600519"     # 双引号
name = '贵州茅台'      # 单引号
```

**建议**: 团队统一选择一种（PEP8 推荐双引号）。

---

### 14. 类型注解不完整

部分函数缺少返回类型注解：
```python
def screen_stocks(self, ...) -> list:       # 应为 -> List[Dict[str, Any]]
def get_realtime_quote(self, symbol) -> dict:  # 应为 -> Dict[str, Any]
```

---

### 15. 日志级别不当

```python
logger.debug(f"腾讯行情解析失败: {e}")   # 解析失败应该用 warning 或 error
logger.debug(f"Baostock K线获取失败")     # 数据获取失败应该用 warning
```

---

## 六、✅ 项目亮点

1. **架构设计合理**: 数据层 / 分析层 / 策略层 / 回测层 分层清晰
2. **多数据源降级**: 腾讯→AKShare、Baostock→AKShare 的 fallback 机制很好
3. **策略模式应用得当**: 6 种交易策略统一接口，扩展方便
4. **文档完善**: README 写得非常详细，包含使用示例和架构图
5. **MCP 协议支持**: 对接 AI Agent 的设计思路先进
6. **六维分析体系**: 技术面/估值/资金面/基本面/财报质量/舆情情绪 覆盖全面

---

## 七、📊 问题统计

| 严重程度 | 数量 | 占比 |
|----------|------|------|
| 🔴 严重 | 4 | 18% |
| 🟡 中等 | 5 | 23% |
| 🔵 安全 | 3 | 14% |
| 🟢 风格 | 3 | 14% |
| ✅ 亮点 | 6 | 27% |
| **合计** | **21** | 100% |

---

## 八、🎯 优先修复建议

1. **立即修复**: 删除 `data_provider.py` 中的死代码（第 102-113 行）
2. **本周完成**: 修复 `backtest.py` 重复导入
3. **近期优化**: 重构 `screen_stocks` 的 N+1 查询问题
4. **持续改进**: 统一异常处理策略、添加缓存上限、提取魔法数字

---

## 九、总结

这是一个**功能丰富、架构清晰**的 A 股分析工具项目，核心分析逻辑扎实，MCP Server 和 CLI 双模式的设计也很实用。主要问题集中在**代码细节规范性**和**性能优化**方面，没有发现致命的逻辑错误或安全漏洞。按照上述建议逐步改进后，代码质量可以达到 **A 级**水平。

> ⚠️ **免责声明**: 本审查仅基于静态代码分析，不保证发现所有问题。建议结合单元测试和集成测试进一步验证。
