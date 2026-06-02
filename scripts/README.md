# Alpha158 A 股选股研究

基于 vnpy Alpha 框架,用 Alpha158 因子 + LightGBM 在 A 股做选股交易研究。
数据来自 stock_new 的 DuckDB(`/home/oracle/stock_new/data/db/market.duckdb`,日线 OHLCV + 基本面,更新至 2026-05)。

## TL;DR — 最终结论

经过最严格口径(消除未来函数 + 跨 regime 滚动 OOS 验证)的研究,**最优方案与天花板**:

> **中盘 CSI500P + 纯 Alpha158 + T+1 开盘成交 + 涨停过滤**
> 3.5 年(2023-2026)滚动 OOS:超额沪深300 **+34.7%**、年化 **17.8%**、**Sharpe 0.78**、最大回撤 -28%

四个改进方向全部验证过,**Sharpe ~0.78 是这套框架在现有数据下的天花板**:

| 方向 | Sharpe | 超额 | 回撤 | 结论 |
|---|---|---|---|---|
| 大盘 universe (MKTCAP500) | 0.72 | +24.3% | -27% | 基准 |
| **换中盘 universe (CSI500P)** | **0.78** | **+34.7%** | -28% | ✅ **全局最优** |
| 中盘 + MA60 择时 | 0.71 | +19.1% | -17% | 控回撤但降 Sharpe |
| 中盘 + 基本面因子 | 0.68 | +33.7% | -36% | 引入噪声,变差 |

**瓶颈在数据信息含量**,不在模型/universe/后处理。再突破须当前没有的数据(精细财务因子、分析师预期、资金流、另类数据)。

## 研究历程 — 一路挤水分

这是本研究最有价值的部分:从"看起来很强"逐步逼近"真实可信"。

| 阶段 | 关键操作 | 超额 | Sharpe |
|---|---|---|---|
| 初始回测 | (含未来函数) | +72% | 2.09 |
| T+1 真实 | signal 平移到次日开盘成交 | +52% | 1.79(单窗口) |
| 滚动 OOS | 纳入 2023(模型反向年) | +24~35% | **0.72~0.78** |

两个关键发现:
1. **未来函数**:vnpy 引擎用 `min(order_price, open)` 开盘价成交,原 signal 基于当日收盘因子却在当日开盘成交 → 用未来信息。修复:`shift_signal_t1` 平移到次日。
2. **regime 依赖**:单看 2025-2026 窗口 Sharpe 1.79,纳入 2023(IC 反向)滚动后骤降到 0.72-0.78。**单窗口会骗人,必须滚动验证**。

## Pipeline 脚本

按执行顺序:

| 脚本 | 作用 |
|---|---|
| `alpha158_bridge.py` | 数据桥:stock_new DuckDB → AlphaLab daily parquet(5770 只,turnover 复权对齐) |
| `build_universe.py` | 构建动态 universe(市值排名区间月度成分股)+ 涨跌停标记 + 沪深300 benchmark |
| `export_fundamental.py` | 导出基本面因子(pe/pb/换手率/量比) |
| `alpha158_train.py` | 分批因子计算 + LightGBM 训练(支持 `--fundamental`) |
| `alpha158_analyze.py` | 因子分析(LightGBM 重要性 + OOS 截面 IC/ICIR) |
| `alpha158_rolling.py` | 滚动 OOS 验证(walk-forward,复用因子矩阵秒级重训) |
| `alpha158_backtest.py` | 选股回测(T+1 / 涨停过滤 / 市场择时 / 沪深300 超额) |
| `robust_strategy.py` | EquityDemoStrategy 子类,修复停牌日买入 KeyError |

## 复现流程

```bash
source /home/oracle/vnpy-venv/bin/activate

# 1. 数据桥(首次,约 16s)
python scripts/alpha158_bridge.py

# 2. 构建 universe + benchmark
python scripts/build_universe.py --name MKTCAP500 --rank-start 1 --rank-end 500           # 大盘
python scripts/build_universe.py --name CSI500P --rank-start 301 --rank-end 800 --skip-aux # 中盘(最优)

# 3. 训练(中盘,12核 workers=6 约 28 分钟)
python scripts/alpha158_train.py --name a158_csi500p --universe CSI500P --batch-size 200 --workers 6

# 4. 滚动 OOS 验证(秒级,复用因子矩阵)
python scripts/alpha158_rolling.py --base a158_csi500p

# 5. 真实回测(最优配置)
python scripts/alpha158_backtest.py --name a158_csi500p_rolling \
    --start 2023-01-01 --end 2026-05-29 --t1 --filter-limit
```

## 方法论铁律(踩坑总结)

1. **回测必加 `--t1`**:引擎用开盘价成交,signal 必须平移到次日,否则未来函数使结果虚高(实测 72% vs 真实 24%)。
2. **评估必用滚动 OOS**:单窗口掩盖 regime 风险(2025-2026 单看 Sharpe 1.79,跨周期真实 0.78)。
3. **universe 用市值动态筛选**:每个调仓日取当时市值排名,零幸存者偏差,无需指数成分股历史名单。
4. **分批因子计算**:Alpha158 纯时序算子,按股票分批与全量数学等价(已对拍 max_abs_diff=0),根治 `prepare_data` 的 OOM。
5. **mktcap 而非 float_cap**:stock_new 的 float_cap 全表缺失,用总市值排名。

## 环境

- venv: `/home/oracle/vnpy-venv`(polars / lightgbm / alphalens / duckdb 1.5.2)
- 产物目录 `lab_data/`、日志 `logs/` 均不入库(.gitignore)
- 核心模块 `vnpy/alpha/` 禁改,所有自定义代码在 `scripts/`

## 未尽方向(若要突破 0.78)

需要 stock_new 现有数据之外的信息源:
- 精细财务因子(成长/质量/盈利趋势,需 `financial_report` 表 + 严格 point-in-time 防未来函数)
- 分析师一致预期、资金流向、另类数据
- regime 自适应建模(2023 反向是核心痛点,简单 MA 择时无效)
