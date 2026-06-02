"""
Alpha158 LightGBM 训练流程(分批因子计算 + 动态 universe)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_train.py                      # MKTCAP500 全量
    python scripts/alpha158_train.py --limit 100          # 只用前 N 只(测试)
    python scripts/alpha158_train.py --batch-size 100     # 内存紧张时减小批

时间窗口(选股交易，OOS=当前市场)：
    IS:    2018-01-01 ~ 2023-12-31
    Valid: 2024-01-01 ~ 2024-12-31
    OOS:   2025-01-01 ~ 2026-05-29

分批原理：Alpha158 是 100% 时序算子，逐股独立，按股票分批计算因子与全量等价。
唯一的截面操作(label zscore)放在各批合并之后执行，保证截面完整。
分批根治 prepare_data 的 OOM —— spawn 模式每个因子 task 会 pickle 整份 df，
减小单批 df 行数即可控制内存。
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import process_cs_norm, process_drop_na
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.utility import Segment
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.trader.constant import Interval

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"

TRAIN_PERIOD = ("2018-01-01", "2023-12-31")
VALID_PERIOD = ("2024-01-01", "2024-12-31")
TEST_PERIOD = ("2025-01-01", "2026-05-29")


def compute_dataset_batched(
    lab: AlphaLab,
    symbols: list[str],
    filters: dict,
    extended_days: int,
    batch_size: int,
    workers: int,
) -> Alpha158:
    """
    分批计算 Alpha158 因子并重组成单个 dataset。

    每批独立 load_bar_df + prepare_data(workers=1)，收集 result_df / raw_df，
    最后 concat 合并、统一加 processor 并 process_data。
    """
    start, end = TRAIN_PERIOD[0], TEST_PERIOD[1]
    periods = dict(train_period=TRAIN_PERIOD, valid_period=VALID_PERIOD, test_period=TEST_PERIOD)

    result_dfs: list[pl.DataFrame] = []
    raw_dfs: list[pl.DataFrame] = []
    last_df: pl.DataFrame | None = None

    n_batches = (len(symbols) + batch_size - 1) // batch_size
    t0 = time.monotonic()

    for bi in range(n_batches):
        batch = symbols[bi * batch_size:(bi + 1) * batch_size]
        df = lab.load_bar_df(batch, Interval.DAILY, start, end, extended_days)
        if df is None or df.is_empty():
            print(f"  批 {bi+1}/{n_batches}: 空数据，跳过", flush=True)
            continue
        last_df = df

        ds = Alpha158(df, **periods)
        batch_filters = {s: filters[s] for s in batch if s in filters}
        ds.prepare_data(filters=batch_filters or None, max_workers=workers)

        result_dfs.append(ds.result_df)
        raw_dfs.append(ds.raw_df)
        elapsed = time.monotonic() - t0
        print(f"  批 {bi+1}/{n_batches}: {len(batch)} 只 | result{tuple(ds.result_df.shape)} "
              f"raw{tuple(ds.raw_df.shape)} | 累计 {elapsed:.0f}s", flush=True)

    if not raw_dfs:
        print("ERROR: 所有批次均为空")
        sys.exit(1)

    # 合并所有批
    merged_result = pl.concat(result_dfs).sort(["datetime", "vt_symbol"])
    merged_raw = pl.concat(raw_dfs).sort(["datetime", "vt_symbol"])
    print(f"合并: result{tuple(merged_result.shape)} raw{tuple(merged_raw.shape)}")

    # 用最后一批 df 构造壳实例，随即覆盖数据字段(Alpha158.__init__ 不计算因子)
    shell = Alpha158(last_df, **periods)
    shell.result_df = merged_result
    shell.raw_df = merged_raw
    shell.infer_df = merged_raw
    shell.learn_df = merged_raw

    # 合并后统一加 processor 并处理(label 截面 zscore 需要完整截面)
    shell.add_processor("learn", partial(process_drop_na, names=["label"]))
    shell.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
    shell.process_data()

    return shell


def add_fundamental_features(dataset: Alpha158, lab_path: Path) -> None:
    """
    把基本面因子 join 到 dataset 的 raw_df/infer_df/learn_df，作为额外特征。

    pe/pb 取倒数(ep=1/pe 盈利收益率、bp=1/pb)以线性化并合理处理负值；
    全部按日截面 rank 到 [0,1]，对极端值稳健、量级统一。缺失填 0.5(中性)。
    label 重排到末尾，保证 LgbModel 的 columns[2:-1] 取到全部因子。
    """
    fund = pl.read_parquet(lab_path / "fundamental.parquet")
    fund = fund.with_columns(
        pl.when(pl.col("pe").abs() > 1e-6).then(1.0 / pl.col("pe")).otherwise(None).alias("ep"),
        pl.when(pl.col("pb").abs() > 1e-6).then(1.0 / pl.col("pb")).otherwise(None).alias("bp"),
    )

    feat_names: list[str] = []
    for c in ["ep", "bp", "turnover_rate", "vol_ratio"]:
        name = f"f_{c}"
        fund = fund.with_columns(
            (pl.col(c).rank() / pl.col(c).count()).over("datetime").alias(name)
        )
        feat_names.append(name)

    fund_feat = fund.select(["datetime", "vt_symbol", *feat_names])

    for attr in ["raw_df", "infer_df", "learn_df"]:
        df: pl.DataFrame = getattr(dataset, attr)
        df = df.join(fund_feat, on=["datetime", "vt_symbol"], how="left")
        df = df.with_columns([pl.col(n).fill_null(0.5) for n in feat_names])
        cols = [c for c in df.columns if c != "label"] + ["label"]
        setattr(dataset, attr, df.select(cols))

    print(f"加入 {len(feat_names)} 个基本面因子: {feat_names}")


def quick_oos_ic(dataset: Alpha158, signal: pl.DataFrame) -> None:
    """OOS 快速 rank IC(轻量、可日志化；完整因子分析见 alpha158_analyze.py)"""
    from scipy.stats import spearmanr

    learn = dataset.fetch_learn(Segment.TEST).select(["datetime", "vt_symbol", "label"])
    merged = signal.join(learn, on=["datetime", "vt_symbol"], how="inner").drop_nulls()

    ics: list[float] = []
    for _key, grp in merged.group_by("datetime"):
        x = grp["signal"].to_numpy()
        y = grp["label"].to_numpy()
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() >= 5:
            ic, _ = spearmanr(x[m], y[m])
            if not np.isnan(ic):
                ics.append(ic)

    if not ics:
        print("OOS rank IC: 无足够样本")
        return

    arr = np.array(ics)
    icir = arr.mean() / (arr.std() + 1e-12)
    print(f"OOS rank IC: mean={arr.mean():+.4f}  std={arr.std():.4f}  "
          f"ICIR={icir:+.3f}  IC>0={np.mean(arr > 0):.1%}  天数={len(arr)}")


def run(name: str, universe: str, limit: int | None, batch_size: int, workers: int,
        fundamental: bool = False, extended_days: int = 100) -> None:
    lab = AlphaLab(str(LAB_PATH))

    start, end = TRAIN_PERIOD[0], TEST_PERIOD[1]
    symbols = lab.load_component_symbols(universe, start, end)
    if limit:
        symbols = sorted(symbols)[:limit]
    filters = lab.load_component_filters(universe, start, end)

    print(f"universe={universe}  股票池: {len(symbols)} 只  batch_size={batch_size}")
    print(f"IS {TRAIN_PERIOD[0]}~{TRAIN_PERIOD[1]} | Valid {VALID_PERIOD[0]}~{VALID_PERIOD[1]} | "
          f"OOS {TEST_PERIOD[0]}~{TEST_PERIOD[1]}")

    # 分批因子计算 + 数据集组装
    t1 = time.monotonic()
    dataset = compute_dataset_batched(lab, symbols, filters, extended_days, batch_size, workers)
    print(f"因子计算+组装完成，耗时 {time.monotonic()-t1:.1f}s")

    if fundamental:
        add_fundamental_features(dataset, LAB_PATH)

    lab.save_dataset(name, dataset)
    print(f"数据集已保存: {name}")

    # 训练 LightGBM
    print("\n--- 训练 LightGBM ---")
    t2 = time.monotonic()
    model = LgbModel(seed=42)
    model.fit(dataset)
    print(f"训练完成，耗时 {time.monotonic()-t2:.1f}s")
    lab.save_model(name, model)
    print(f"模型已保存: {name}")

    # 测试集预测 + 信号绩效
    print("\n--- OOS 信号绩效 ---")
    pre = model.predict(dataset, Segment.TEST)
    df_t = dataset.fetch_infer(Segment.TEST)
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal = df_t["datetime", "vt_symbol", "signal"]

    lab.save_signal(name, signal)
    print(f"信号已保存: {name}  ({len(signal):,} 行)")

    quick_oos_ic(dataset, signal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha158 LightGBM 训练(分批+universe)")
    parser.add_argument("--name", default="a158_mktcap500", help="实验名称")
    parser.add_argument("--universe", default="MKTCAP500", help="AlphaLab 虚拟指数名")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数(测试用)")
    parser.add_argument("--batch-size", type=int, default=200, help="分批大小")
    parser.add_argument("--workers", type=int, default=1, help="prepare_data 并行进程数")
    parser.add_argument("--fundamental", action="store_true", help="加入基本面因子(ep/bp/换手率/量比)")
    args = parser.parse_args()

    run(
        name=args.name,
        universe=args.universe,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        fundamental=args.fundamental,
    )


if __name__ == "__main__":
    main()
