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
TEST_PERIOD = ("2025-01-01", "2026-06-04")


def compute_dataset_batched(
    lab: AlphaLab,
    symbols: list[str],
    filters: dict,
    extended_days: int,
    batch_size: int,
    workers: int,
    label_fwd: int = 2,
) -> Alpha158:
    """
    分批计算 Alpha158 因子并重组成单个 dataset。

    每批算完立即写临时 parquet 并释放内存（避免全批攒在 RAM 导致 OOM）；
    最后用 scan_parquet 流式合并，峰值内存 ≈ 1批 + 1个完整 DataFrame。
    """
    import shutil
    import tempfile

    start, end = TRAIN_PERIOD[0], TEST_PERIOD[1]
    periods = dict(train_period=TRAIN_PERIOD, valid_period=VALID_PERIOD, test_period=TEST_PERIOD)

    import gc
    tmpdir = Path(tempfile.mkdtemp(prefix="a158_batch_"))
    raw_paths: list[str] = []
    last_df: pl.DataFrame | None = None

    n_batches = (len(symbols) + batch_size - 1) // batch_size
    t0 = time.monotonic()

    try:
        for bi in range(n_batches):
            batch = symbols[bi * batch_size:(bi + 1) * batch_size]
            df = lab.load_bar_df(batch, Interval.DAILY, start, end, extended_days)
            if df is None or df.is_empty():
                print(f"  批 {bi+1}/{n_batches}: 空数据，跳过", flush=True)
                continue
            last_df = df

            ds = Alpha158(df, **periods)
            if label_fwd != 2:
                ds.set_label(f"ts_delay(close, -{label_fwd + 1}) / ts_delay(close, -1) - 1")
            batch_filters = {s: filters[s] for s in batch if s in filters}
            ds.prepare_data(filters=batch_filters or None, max_workers=workers)

            # 只写 raw_df（~72MB/批，特征列子集），跳过 result_df（~165MB/批）节省磁盘和内存
            rawp = str(tmpdir / f"raw_{bi:03d}.parquet")
            ds.raw_df.write_parquet(rawp)
            raw_paths.append(rawp)

            elapsed = time.monotonic() - t0
            print(f"  批 {bi+1}/{n_batches}: {len(batch)} 只 | raw{tuple(ds.raw_df.shape)} | 累计 {elapsed:.0f}s", flush=True)

            # 立即释放当批内存
            del ds, df
            gc.collect()

        if not raw_paths:
            print("ERROR: 所有批次均为空")
            sys.exit(1)

        # 只合并 raw_df（~1.2GB），跳过 result_df（~3GB）避免 OOM
        print("合并所有批次 raw_df（从磁盘流式读取）...", flush=True)
        merged_raw = pl.scan_parquet(raw_paths).sort(["datetime", "vt_symbol"]).collect()
        print(f"合并: raw{tuple(merged_raw.shape)}", flush=True)

    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)

    # 用最后一批 df 构造壳实例，随即覆盖数据字段(Alpha158.__init__ 不计算因子)
    shell = Alpha158(last_df, **periods)
    shell.result_df = None   # 跳过 ~3GB result_df：infer/learn 只需 raw_df
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

    for attr in ["infer_df", "learn_df"]:  # 跳过 raw_df：节省一份 ~1.2GB 副本
        df: pl.DataFrame = getattr(dataset, attr)
        if df is None:
            continue
        df = df.join(fund_feat, on=["datetime", "vt_symbol"], how="left")
        df = df.with_columns([pl.col(n).fill_null(0.5) for n in feat_names])
        cols = [c for c in df.columns if c != "label"] + ["label"]
        setattr(dataset, attr, df.select(cols))

    print(f"加入 {len(feat_names)} 个基本面因子: {feat_names}")


def add_industry_features(dataset: Alpha158, lab_path: Path) -> None:
    """
    将申万 L1 行业编码加入 LightGBM 特征（整数 label encoding，0~30）。
    让模型学习行业内超额收益，而非跨行业的绝对排名。
    行业 ID 作为普通数值特征，LightGBM 自动找最优行业分割边界。
    """
    ind_path = lab_path / "industry.parquet"
    if not ind_path.exists():
        print("⚠️  industry.parquet 不存在，跳过行业特征（先运行 export_industry.py）")
        return

    industry = pl.read_parquet(str(ind_path)).select(["vt_symbol", "industry_id"])

    for attr in ["infer_df", "learn_df"]:  # 跳过 raw_df：节省一份 ~5GB 副本
        df: pl.DataFrame = getattr(dataset, attr)
        if df is None:
            continue
        df = df.join(industry, on="vt_symbol", how="left")
        df = df.with_columns(pl.col("industry_id").fill_null(-1).cast(pl.Float64))
        cols = [c for c in df.columns if c != "label"] + ["label"]
        setattr(dataset, attr, df.select(cols))

    n_industries = industry["industry_id"].n_unique()
    print(f"加入行业特征: industry_id (0~{n_industries-1}，31 个申万 L1)")


def add_financial_features(dataset: Alpha158, lab_path: Path) -> None:
    """
    把 point-in-time 财务因子(成长/质量)join 到 dataset，作为额外特征。

    10 个因子(净利增速/营收增速/ROE/毛利率/净利率/资产负债率/每股经营现金流
    + delta_roe(ROE同比变化) + cfo_quality(OCF/EPS应计质量) + rev_accel(营收增速加速度))
    按日截面 rank 到 [0,1]，方向由 LightGBM 自学。缺失填 0.5。
    """
    fin = pl.read_parquet(lab_path / "financials.parquet")
    raw_cols = ["np_yoy", "rev_yoy", "roe", "gross_margin", "net_margin", "debt_ratio", "ocf_ps",
                "delta_roe", "cfo_quality", "rev_accel", "sue", "rev_sue"]

    feat_names: list[str] = []
    for c in raw_cols:
        name = f"fin_{c}"
        fin = fin.with_columns(
            (pl.col(c).rank() / pl.col(c).count()).over("datetime").alias(name)
        )
        feat_names.append(name)

    fin_feat = fin.select(["datetime", "vt_symbol", *feat_names])

    for attr in ["infer_df", "learn_df"]:  # 跳过 raw_df：节省一份 ~5GB 副本
        df: pl.DataFrame = getattr(dataset, attr)
        if df is None:
            continue
        df = df.join(fin_feat, on=["datetime", "vt_symbol"], how="left")
        df = df.with_columns([pl.col(n).fill_null(0.5) for n in feat_names])
        cols = [c for c in df.columns if c != "label"] + ["label"]
        setattr(dataset, attr, df.select(cols))

    print(f"加入 {len(feat_names)} 个财务因子: {feat_names}")


def add_dividend_features(dataset: Alpha158, lab_path: Path) -> None:
    """
    TTM 股息率因子(pytdx xdxr，point-in-time)。
    按日截面 rank 到 [0,1]，缺失填 0.5。文件不存在时静默跳过。
    """
    div_path = lab_path / "dividend.parquet"
    if not div_path.exists():
        print("⚠️  dividend.parquet 不存在，跳过股息率特征（先运行 export_dividend.py）")
        return

    div = pl.read_parquet(str(div_path))
    feat_name = "fin_div_yield"
    div = div.with_columns(
        (pl.col("div_yield_ttm").rank() / pl.col("div_yield_ttm").count()).over("datetime").alias(feat_name)
    )
    div_feat = div.select(["datetime", "vt_symbol", feat_name])

    for attr in ["infer_df", "learn_df"]:
        df: pl.DataFrame = getattr(dataset, attr)
        if df is None:
            continue
        df = df.join(div_feat, on=["datetime", "vt_symbol"], how="left")
        df = df.with_columns(pl.col(feat_name).fill_null(0.5))
        cols = [c for c in df.columns if c != "label"] + ["label"]
        setattr(dataset, attr, df.select(cols))

    print(f"加入股息率因子: [{feat_name}]")


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
        fundamental: bool = False, financial: bool = False, industry: bool = False,
        label_fwd: int = 2, extended_days: int = 100) -> None:
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
    dataset = compute_dataset_batched(lab, symbols, filters, extended_days, batch_size, workers, label_fwd)
    print(f"因子计算+组装完成，耗时 {time.monotonic()-t1:.1f}s")

    if fundamental:
        add_fundamental_features(dataset, LAB_PATH)
    if financial:
        add_financial_features(dataset, LAB_PATH)
    if industry:
        add_industry_features(dataset, LAB_PATH)

    # 释放不参与训练的大 DataFrame，降低 pkl 体积和 LGB 峰值内存
    # result_df / raw_df 只在因子计算中间态用到，训练/推理/rolling 不需要
    import gc
    dataset.result_df = None
    dataset.raw_df = None
    gc.collect()

    lab.save_dataset(name, dataset)
    print(f"数据集已保存: {name}")

    # 再次 GC，确保 save 过程中的临时对象被回收
    gc.collect()

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
    parser.add_argument("--financial", action="store_true", help="加入财务因子(成长/质量，point-in-time)")
    parser.add_argument("--industry", action="store_true", help="加入申万 L1 行业特征(行业内 alpha 学习)")
    parser.add_argument("--hold", type=int, default=2, help="持有天数(label 周期)，默认 2 日")
    args = parser.parse_args()

    run(
        name=args.name,
        universe=args.universe,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        fundamental=args.fundamental,
        financial=args.financial,
        industry=args.industry,
        label_fwd=args.hold,
    )


if __name__ == "__main__":
    main()
