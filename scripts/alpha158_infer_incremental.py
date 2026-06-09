"""
增量推理：追加新交易日因子到 infer_df，无需重建全量数据集。

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_infer_incremental.py           # 自动检测新日期
    python scripts/alpha158_infer_incremental.py --skip-bridge  # 跳过数据桥(已手动更新)

原理：
    Alpha158 时序算子最长回看 ~60 交易日。只需加载 last_date-LOOKBACK_BUFFER 到
    new_end 的 OHLCV 数据，计算因子后过滤出 > last_date 的新行，追加到 infer_df。
    learn_df(2018-2025 有标签数据)和模型权重完全不变，秒级完成。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.trader.constant import Interval

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
ROOT = Path(__file__).resolve().parents[1]
PY = "/home/oracle/vnpy-venv/bin/python"

DATASET_NAME = "a158_csi500_fin"
UNIVERSE = "CSI500"
LOOKBACK_BUFFER = 90   # 缓冲交易日数，确保 60日回看窗有足够历史
EXTENDED_DAYS = 100    # lab.load_bar_df 往 buffer_start 前额外延伸的天数

TRAIN_PERIOD = ("2018-01-01", "2023-12-31")
VALID_PERIOD = ("2024-01-01", "2024-12-31")


# ─── 财务因子（与 alpha158_train.py 保持一致）────────────────────────────────

def compute_financial_features(lab_path: Path) -> pl.DataFrame:
    fin = pl.read_parquet(lab_path / "financials.parquet")
    raw_cols = ["np_yoy", "rev_yoy", "roe", "gross_margin", "net_margin", "debt_ratio", "ocf_ps",
                "delta_roe", "cfo_quality", "rev_accel", "sue", "rev_sue"]
    for c in raw_cols:
        name = f"fin_{c}"
        fin = fin.with_columns(
            (pl.col(c).rank() / pl.col(c).count()).over("datetime").alias(name)
        )
    feat_names = [f"fin_{c}" for c in raw_cols]
    return fin.select(["datetime", "vt_symbol", *feat_names])


def compute_dividend_features(lab_path: Path) -> pl.DataFrame | None:
    div_path = lab_path / "dividend.parquet"
    if not div_path.exists():
        return None
    div = pl.read_parquet(str(div_path))
    div = div.with_columns(
        (pl.col("div_yield_ttm").rank() / pl.col("div_yield_ttm").count()).over("datetime").alias("fin_div_yield")
    )
    return div.select(["datetime", "vt_symbol", "fin_div_yield"])


# ─── 增量因子计算 ─────────────────────────────────────────────────────────────

def compute_incremental_infer(
    lab: AlphaLab,
    symbols: list[str],
    filters: dict,
    buffer_start: str,
    new_end: str,
    last_date,
    workers: int,
) -> pl.DataFrame:
    """
    计算 buffer_start~new_end 的 Alpha158 因子，只返回 > last_date 的新行。
    数据量小（~90交易日 × 1148只），无需分批，一次性处理。
    """
    periods = dict(
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period=(buffer_start, new_end),
    )

    print(f"加载 OHLCV: {buffer_start} ~ {new_end}，{len(symbols)} 只", flush=True)
    df = lab.load_bar_df(symbols, Interval.DAILY, buffer_start, new_end, EXTENDED_DAYS)
    if df is None or df.is_empty():
        print("ERROR: 无法加载 OHLCV 数据")
        sys.exit(1)
    print(f"  OHLCV shape: {df.shape}", flush=True)

    ds = Alpha158(df, **periods)
    ds.prepare_data(filters=filters or None, max_workers=workers)

    # infer_df = raw_df（未归一化特征，与 train.py 保持一致）
    new_rows = ds.raw_df.filter(pl.col("datetime") > last_date)
    print(f"  新日期行数: {len(new_rows)}（{new_rows['datetime'].min()} ~ {new_rows['datetime'].max()}）", flush=True)
    return new_rows


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha158 增量推理：追加新交易日因子")
    parser.add_argument("--name", default=DATASET_NAME, help="dataset 名称")
    parser.add_argument("--universe", default=UNIVERSE, help="universe 名称")
    parser.add_argument("--workers", type=int, default=4, help="prepare_data 并行进程数")
    parser.add_argument("--skip-bridge", action="store_true", help="跳过数据桥步骤")
    parser.add_argument("--skip-rolling", action="store_true", help="只更新 dataset，不重跑滚动推理")
    args = parser.parse_args()

    t0 = time.monotonic()

    # ── Step 1: 日线数据更新 ────────────────────────────────────────────────
    if not args.skip_bridge:
        print("\n=== Step 1: Tushare 日线增量更新 ===", flush=True)
        r = subprocess.run([PY, "scripts/fetch_daily_tushare.py"], cwd=str(ROOT))
        if r.returncode != 0:
            print("ERROR: Tushare 日线更新失败")
            sys.exit(1)
        print(f"日线更新完成，耗时 {time.monotonic()-t0:.0f}s", flush=True)
    else:
        print("跳过日线更新（--skip-bridge）", flush=True)

    # ── Step 2: 检测新日期 ──────────────────────────────────────────────────
    print("\n=== Step 2: 检测新日期 ===", flush=True)
    lab = AlphaLab(str(LAB_PATH))
    dataset = lab.load_dataset(args.name)
    if dataset is None:
        print(f"ERROR: 找不到 dataset {args.name}，请先运行 alpha158_train.py")
        sys.exit(1)

    last_date = dataset.infer_df["datetime"].max()
    last_date_str = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)[:10]
    print(f"现有 infer_df 最新日期: {last_date_str}  ({len(dataset.infer_df):,} 行)", flush=True)

    # 扫描 parquet 文件中的最新可用日期
    daily_dir = LAB_PATH / "daily"
    sample_files = sorted(daily_dir.glob("*.parquet"))[:50]
    max_dates = []
    for f in sample_files:
        try:
            d = pl.read_parquet(f, columns=["datetime"])["datetime"].max()
            if d is not None:
                max_dates.append(d)
        except Exception:
            continue

    if not max_dates:
        print("ERROR: 无法读取 daily parquet 文件")
        sys.exit(1)
    new_end_dt = max(max_dates)
    new_end_str = str(new_end_dt.date()) if hasattr(new_end_dt, "date") else str(new_end_dt)[:10]
    print(f"parquet 最新可用日期: {new_end_str}", flush=True)

    if new_end_dt <= last_date:
        print("已是最新，无需更新。退出。")
        return

    # ── Step 3: 计算增量因子 ────────────────────────────────────────────────
    print(f"\n=== Step 3: 计算增量 Alpha158 因子（新增 {new_end_str} 前的交易日）===", flush=True)

    # buffer_start = last_date 往前 LOOKBACK_BUFFER 个交易日
    all_dates = pl.concat([
        pl.read_parquet(f, columns=["datetime"]) for f in sample_files[:10]
    ])["datetime"].unique().sort()
    idx = (all_dates <= last_date).sum() - 1
    buf_idx = max(0, idx - LOOKBACK_BUFFER)
    buffer_start_str = str(all_dates[buf_idx].date()) if hasattr(all_dates[buf_idx], "date") else str(all_dates[buf_idx])[:10]
    print(f"缓冲起始日: {buffer_start_str}（{LOOKBACK_BUFFER} 交易日缓冲）", flush=True)

    # universe + filters
    symbols = lab.load_component_symbols(args.universe, TRAIN_PERIOD[0], new_end_str)
    filters = lab.load_component_filters(args.universe, TRAIN_PERIOD[0], new_end_str)
    print(f"universe={args.universe}  股票池: {len(symbols)} 只", flush=True)

    t_factor = time.monotonic()
    # 增量推理不需要成员过滤（过滤是为了训练时避免前瞻偏差，推理用全量）
    new_rows = compute_incremental_infer(
        lab, symbols, None, buffer_start_str, new_end_str,
        last_date, args.workers,
    )
    print(f"因子计算完成，耗时 {time.monotonic()-t_factor:.1f}s", flush=True)

    if new_rows.is_empty():
        print("无新行，退出。")
        return

    # ── Step 4: 追加财务因子 + 股息率因子 ────────────────────────────────────
    print("\n=== Step 4: 追加财务因子 ===", flush=True)
    fin_feat = compute_financial_features(LAB_PATH)
    # 只取新行涉及的日期，避免全量 join
    new_dates = new_rows["datetime"].unique()
    fin_new = fin_feat.filter(pl.col("datetime").is_in(new_dates.to_list()))

    new_rows = new_rows.join(fin_new, on=["datetime", "vt_symbol"], how="left")
    fin_cols = [c for c in fin_feat.columns if c not in ("datetime", "vt_symbol")]
    new_rows = new_rows.with_columns([pl.col(n).fill_null(0.5) for n in fin_cols])


    # 列对齐：新行列顺序与现有 infer_df 一致（有 label 列则保留，否则补 NaN）
    existing_cols = dataset.infer_df.columns
    for col in existing_cols:
        if col not in new_rows.columns:
            dtype = dataset.infer_df[col].dtype
            new_rows = new_rows.with_columns(pl.lit(None).cast(dtype).alias(col))
    new_rows = new_rows.select(existing_cols)

    # ── Step 5: 追加到 infer_df，保存 ──────────────────────────────────────
    print("\n=== Step 5: 追加 infer_df 并保存 dataset ===", flush=True)
    old_len = len(dataset.infer_df)
    dataset.infer_df = pl.concat([dataset.infer_df, new_rows]).sort(["datetime", "vt_symbol"])
    print(f"infer_df: {old_len:,} → {len(dataset.infer_df):,} 行（+{len(new_rows):,}）", flush=True)

    # learn_df 不更新（新日期无标签，训练数据不变）

    # 更新数据集 test_period 终止日
    from vnpy.alpha.dataset.utility import Segment
    train_p, valid_p, test_p = (
        dataset.data_periods[Segment.TRAIN],
        dataset.data_periods[Segment.VALID],
        dataset.data_periods[Segment.TEST],
    )
    dataset.data_periods[Segment.TEST] = (test_p[0], new_end_str)
    print(f"dataset test_period 更新: {test_p[1]} → {new_end_str}", flush=True)

    lab.save_dataset(args.name, dataset)
    print(f"dataset 已保存: {args.name}", flush=True)

    if args.skip_rolling:
        print("跳过滚动推理（--skip-rolling）")
        return

    # ── Step 6: 滚动推理 ───────────────────────────────────────────────────
    print(f"\n=== Step 6: 滚动推理（W4 终止 {new_end_str}）===", flush=True)
    r = subprocess.run(
        [PY, "scripts/alpha158_rolling.py", "--base", args.name, "--w4-end", new_end_str],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        print("ERROR: 滚动推理失败")
        sys.exit(1)

    # ── Step 7: 生成选股清单 ───────────────────────────────────────────────
    print("\n=== Step 7: 生成选股清单 ===", flush=True)
    r = subprocess.run(
        [PY, "scripts/generate_signal.py", "--top-k", "50"],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        print("ERROR: 信号生成失败")
        sys.exit(1)

    print(f"\n✓ 增量推理完成，总耗时 {(time.monotonic()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
