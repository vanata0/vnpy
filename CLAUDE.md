# CLAUDE.md — vnpy Alpha 研究项目

## 项目定位

这是 [vnpy/vnpy](https://github.com/vnpy/vnpy) 的个人 fork，**仅用于 Alpha158 + ML 量化研究**，不做实盘交易接入。

核心目标：
1. 从 `stock_new` 项目的 DuckDB 导入 A 股 OHLCV 数据（数据桥）
2. 用 Alpha158 计算 158 个技术因子
3. 训练 LightGBM 模型，验证 A 股 alpha 有效性
4. 用 alphalens 生成因子分析报告

**与 stock_new 的关系**：计算完全隔离在本项目，结果（每日信号）可选择性写回 stock_new 的 DuckDB，stock_new 的 18 个规则策略不受影响。

## 目录结构

```
vnpy/
├── vnpy/alpha/              # 核心模块（不要改）
│   ├── dataset/
│   │   ├── datasets/
│   │   │   ├── alpha_158.py     # 158 个因子定义（来自 Qlib）
│   │   │   └── alpha_101.py     # 101 个 WorldQuant 因子
│   │   ├── ts_function.py       # 时序算子（ts_delay/ts_mean/ts_std...）
│   │   ├── cs_function.py       # 截面算子（cs_rank...）
│   │   └── template.py          # AlphaDataset 基类
│   ├── model/
│   │   └── models/
│   │       ├── lgb_model.py     # LightGBM 封装
│   │       ├── mlp_model.py     # MLP 封装
│   │       └── lasso_model.py   # Lasso 封装
│   ├── strategy/
│   │   └── strategies/
│   │       └── equity_demo_strategy.py  # 持仓管理策略（top_k/换仓/成本）
│   └── lab.py                   # AlphaLab：数据管理（parquet 读写）
├── examples/alpha_research/     # 官方 notebook 示例（参考用）
│   ├── research_workflow_lgb.ipynb   # LightGBM 完整流程参考
│   └── research_workflow_alpha101.ipynb
├── scripts/                     # 本项目自定义脚本（需新建）
│   └── alpha158_bridge.py       # 数据桥：stock_new DuckDB → AlphaLab parquet
├── lab_data/                    # AlphaLab 数据目录（运行后生成）
│   ├── daily/                   # 每股日线 parquet（SH.600001.parquet...）
│   └── ...
└── tests/
    └── test_alpha101.py         # 因子测试（参考格式）
```

## 运行环境

```bash
# 激活 venv（含 polars/lightgbm/alphalens/scipy）
source /home/oracle/vnpy-venv/bin/activate

# 验证环境
python -c "import polars, lightgbm, alphalens; print('OK')"
```

**不要用系统 Python**，alphalens/lightgbm 只在 vnpy-venv 里。

## 数据桥规格（stock_new → vnpy）

`scripts/alpha158_bridge.py` 是整个项目的入口，数据格式要求：

| vnpy 字段 | stock_new 来源 | 备注 |
|-----------|---------------|------|
| `datetime` | `stock_daily.date` | 转 datetime |
| `vt_symbol` | `stock_daily.code` | 格式转换见下 |
| `open/high/low/close` | 直接对应 | 前复权用 `*_qfq` 字段 |
| `volume` | `stock_daily.volume` | 直接对应 |
| `vwap` | `amount / volume` | 无精确 vwap，估算 |
| `turnover` | `stock_daily.amount` | 成交额 |
| `open_interest` | 0 | 股票无需此字段 |

**symbol 格式转换**：
```python
def to_vt_symbol(code: str) -> str:
    if code.startswith(('8', '4')): return f"BJ.{code}"
    if code.startswith('6'):        return f"SH.{code}"
    return f"SZ.{code}"
```

**stock_new DuckDB 路径**：`/home/oracle/stock_new/data/db/market.duckdb`
注意：stock_new 后端运行时持有写锁，脚本用 `read_only=True` 或先复制文件。

## 开发规范

### 代码规范

- **严禁改动 `vnpy/alpha/` 目录下的核心模块**，所有自定义代码放 `scripts/`
- 脚本用 `vnpy-venv` 的 Python，不依赖 stock_new 的运行环境
- 不要在 `scripts/` 里引入 FastAPI / Vue 相关依赖

### 训练流程规范

每次训练前记录：
- 训练区间（IS）、验证区间（Valid）、测试区间（OOS）
- 关键超参数（num_leaves、learning_rate、n_estimators）
- OOS 核心指标：年化收益、Sharpe、最大回撤

**禁止**：只看 IS 结果就调参，必须在 OOS 验证后再决定是否合并参数变更。

### 数据注意事项

- **停牌日**：volume=0 时 vwap=amount/0，需过滤或填 close
- **涨跌停日**：vwap 估算偏差较大，可接受（不影响整体因子质量）
- **科创板（688）/ 北交所（8/4）**：Alpha158 包含，不需要特别排除（和 stock_new 策略过滤逻辑不同）

### 提交信息

参考格式（与 stock_new 风格一致）：
```
feat(bridge): 数据桥初版 — DuckDB → AlphaLab parquet，5000 只 580 天
fix(alpha158): 停牌日 vwap 除零保护
chore(lab): 增量更新逻辑，只追加新日期数据
```

## 参考资料

- 官方完整流程：`examples/alpha_research/research_workflow_lgb.ipynb`
- Alpha158 因子定义：`vnpy/alpha/dataset/datasets/alpha_158.py`
- AlphaLab 接口：`vnpy/alpha/lab.py`
- alphalens 因子分析：`dataset.show_feature_performance("roc_5")`

## 交互语言

**始终使用中文**与用户交互。
