# Alpha158 A 股选股研究

基于 vnpy Alpha 框架,用 Alpha158 因子 + LightGBM 在 A 股做选股交易研究。
数据来自 stock_new 的 DuckDB(`/home/oracle/stock_new/data/db/market.duckdb`,日线 OHLCV + 基本面 + 财报,更新至 2026-05)。

## TL;DR — 最终结论

经过最严格口径(消除未来函数 + 跨 regime 滚动 OOS 验证)的研究,**最优方案**:

> **中盘 CSI500P + Alpha158 + point-in-time 财务因子 + T+1 开盘成交 + 涨停过滤**
> 3.5 年(2023-2026)滚动 OOS:超额沪深300 **+44.7%**、年化 **20.7%**、**Sharpe 0.85**、最大回撤 -28%、收益回撤比 2.31

**核心洞察:攻因子要选对维度。** 纯量价(Alpha158)Sharpe 天花板 ~0.78,加**成长/质量财务因子**(营收增速/净利增速/ROE)突破到 0.85——这是量价无法隐含的正交信息。

各方向全景:

| 方向 | Sharpe | 超额 | 回撤 | 结论 |
|---|---|---|---|---|
| 大盘 universe (MKTCAP500) | 0.72 | +24.3% | -27% | 基准 |
| 换中盘 universe (CSI500P) | 0.78 | +34.7% | -28% | 纯量价天花板 |
| 中盘 + MA60 择时 | 0.71 | +19.1% | -17% | ❌ 控回撤但降 Sharpe |
| 中盘 + 粗基本面 (pe/pb) | 0.68 | +33.7% | -36% | ❌ 估值已被量价隐含,引噪声 |
| **中盘 + 财务因子 (成长/质量)** | **0.85** | **+44.7%** | -28% | ✅ **全局最优** |

## 研究历程 — 先挤水分,再攻因子

### 第一阶段:挤水分(从虚高到可信)

| 阶段 | 关键操作 | 超额 | Sharpe |
|---|---|---|---|
| 初始回测 | (含未来函数) | +72% | 2.09 |
| T+1 真实 | signal 平移到次日开盘成交 | +52% | 1.79(单窗口) |
| 滚动 OOS | 纳入 2023(模型反向年) | +24~35% | 0.72~0.78 |

两个关键发现:
1. **未来函数**:vnpy 引擎用 `min(order_price, open)` 开盘价成交,原 signal 基于当日收盘因子却在当日开盘成交 → 用未来信息。修复:`shift_signal_t1` 平移到次日。
2. **regime 依赖**:单看 2025-2026 窗口 Sharpe 1.79,纳入 2023(IC 反向)滚动后降到 0.72-0.78。**单窗口会骗人,必须滚动验证**。

### 第二阶段:攻因子(突破天花板)

纯量价撞墙在 Sharpe 0.78 后,尝试加因子:

- **❌ 粗基本面(pe/pb/换手率/量比)**:滚动 IC 几乎无增量,回测 Sharpe 反降到 0.68、回撤恶化 -36%。**估值信息已被量价行为隐含**。
- **✅ point-in-time 财务因子(营收增速/净利增速/ROE/毛利率/净利率/负债率/现金流)**:滚动 IC 提升(0.0247→0.0267),2023 反向最轻,回测 **Sharpe 0.78→0.85**。营收增速在 LightGBM 重要性排第 3。**成长/质量是财报硬数据,股价无法反推,真正正交**。
- 关键技术:用 DuckDB **ASOF JOIN** 按公告日 `notice_date` 匹配,每个交易日只用已公告财报,严格 point-in-time 防未来函数(已验证公告日前后数值正确跳变)。

## Pipeline 脚本

按执行顺序:

| 脚本 | 作用 |
|---|---|
| `alpha158_bridge.py` | 数据桥:stock_new DuckDB → AlphaLab daily parquet(5770 只,turnover 复权对齐) |
| `build_universe.py` | 动态 universe(市值排名区间月度成分股)+ 涨跌停标记 + 沪深300 benchmark |
| `export_fundamental.py` | 粗基本面(pe/pb/换手率/量比)— 已验证无用,保留供参考 |
| `export_financials.py` | **point-in-time 财务因子(成长/质量,ASOF JOIN 防未来函数)— 突破关键** |
| `alpha158_train.py` | 分批因子计算 + LightGBM 训练(`--financial` 加财务因子,`--fundamental` 加粗基本面) |
| `alpha158_analyze.py` | 因子分析(LightGBM 重要性 + OOS 截面 IC/ICIR) |
| `alpha158_rolling.py` | 滚动 OOS 验证(walk-forward,复用因子矩阵秒级重训) |
| `alpha158_backtest.py` | 选股回测(T+1 / 涨停过滤 / 市场择时 / 沪深300 超额) |
| `robust_strategy.py` | EquityDemoStrategy 子类,修复停牌日买入 KeyError |

## 复现流程(最优方案)

```bash
source /home/oracle/vnpy-venv/bin/activate

# 1. 数据桥(首次)
python scripts/alpha158_bridge.py

# 2. universe(中盘)+ benchmark + 财务因子
python scripts/build_universe.py --name CSI500P --rank-start 301 --rank-end 800
python scripts/export_financials.py

# 3. 训练(中盘 + 财务因子,12核 workers=6 约 28 分钟)
python scripts/alpha158_train.py --name a158_csi500p_fin --universe CSI500P \
    --batch-size 200 --workers 6 --financial

# 4. 滚动 OOS 验证(秒级,复用因子矩阵)
python scripts/alpha158_rolling.py --base a158_csi500p_fin

# 5. 真实回测(最优配置 → Sharpe 0.85)
python scripts/alpha158_backtest.py --name a158_csi500p_fin_rolling \
    --start 2023-01-01 --end 2026-05-29 --t1 --filter-limit
```

## 方法论铁律(踩坑总结)

1. **回测必加 `--t1`**:引擎用开盘价成交,signal 必须平移到次日,否则未来函数使结果虚高(实测 72% vs 真实 24%)。
2. **评估必用滚动 OOS**:单窗口掩盖 regime 风险(2025-2026 单看 Sharpe 1.79,跨周期真实 0.78-0.85)。
3. **基本面因子必须 point-in-time**:用公告日 `notice_date`(非报告期)做 ASOF JOIN,否则用了还没公布的财报 = 未来函数。
4. **攻因子选对维度**:估值(pe/pb)无用(已被量价隐含),成长/质量(营收增速/ROE)正交有效。
5. **universe 用市值动态筛选**:每个调仓日取当时市值排名,零幸存者偏差。
6. **分批因子计算**:Alpha158 纯时序算子,按股票分批与全量数学等价(已对拍 max_abs_diff=0),根治 OOM。
7. **mktcap 而非 float_cap**:stock_new 的 float_cap 全表缺失。
8. **看 IC 不看单次 Sharpe**:前复权(`*_qfq`)在新除权时重算全部历史价,全量重建后回测 Sharpe 会漂移(同配置实测 0.85↔0.64),但 IC 稳定(0.043)。Sharpe 真实是区间 **~0.7(0.64-0.85)**,IC 才是模型有效性的可靠指标。漂移不影响出信号(与实盘前复权口径一致),纯研究要可复现就固定 DB 快照。

## 环境

- venv: `/home/oracle/vnpy-venv`(polars / lightgbm / alphalens / duckdb 1.5.2)
- 硬件 12 核 / 31G;分批训练 workers=6,全量 28 分钟
- 产物目录 `lab_data/`、日志 `logs/` 不入库(.gitignore);核心模块 `vnpy/alpha/` 禁改,自定义代码在 `scripts/`

## 下一步:还能试哪些因子组合

**已验证的规律:正交于量价的维度才有效**(成长/质量 ✅,估值 ❌)。按此原则排优先级:

| 优先级 | 因子组合 | 正交逻辑 | 数据 |
|---|---|---|---|
| 🔥 高 | **盈利质量**(应计项 = 净利润 − 经营现金流的背离) | 识别"纸面利润"陷阱,量价无法反推 | ✅ net_profit + oper_cf |
| 🔥 高 | **行业中性化**(现有因子在申万行业内去 β) | 剥离行业 β、提纯选股 α,**不需新数据** | ✅ industry_sw 已有 |
| 中 | **成长二阶**(营收增速的加速度、ROE 趋势) | 捕捉"边际改善中"的公司,比静态增速更前瞻 | ✅ 财报时序 |
| 中 | **GARP**(成长 × 低估值交互项) | 估值单独无效,但成长+合理估值可避开高估值成长陷阱 | ✅ 已有 |
| 低 | 财报 surprise(实际 vs 分析师预期) | 预期差是强 alpha | ❌ 需分析师一致预期数据 |
| 低 | 资金流 / 北向持仓 | 聪明钱信号 | ❌ stock_new 无 |

**最该先试:盈利质量 + 行业中性化** —— 数据现成、正交性强、学界长期验证有效。
**不必再碰**:估值(pe/pb)、MA 择时(已证无效)。
regime 自适应(2023 反向痛点)难度高、简单 MA 已失败,优先级靠后。
