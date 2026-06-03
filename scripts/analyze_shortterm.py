"""
短线因子 IC 扫描 — 找不同持有周期(1/2/5日)下最有效的因子组合

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/analyze_shortterm.py --name a158_csi500p_fin

对 OOS 段，计算每个因子(Alpha158 量价 + 财务)对未来 1日/2日/5日 收益的截面 ICIR，
识别短线(1-2日)最强因子，以及"短线强、中线弱"的短线特有因子。
未来收益均按 T+1 开盘口径：fwd_kd = close[t+1+k]/close[t+1] - 1(无未来函数)。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset.utility import Segment

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"


def daily_icir(merged: pl.DataFrame, col: str, target: str) -> tuple[float, float]:
    """返回 (IC均值, ICIR)"""
    sub = merged.select(["datetime", col, target]).drop_nulls()
    ics: list[float] = []
    for _key, g in sub.group_by("datetime"):
        x = g[col].to_numpy()
        y = g[target].to_numpy()
        m = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if m.sum() >= 10:
            v, _ = spearmanr(x[m], y[m])
            if not np.isnan(v):
                ics.append(v)
    if not ics:
        return 0.0, 0.0
    a = np.array(ics)
    return float(a.mean()), float(a.mean() / (a.std() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="短线因子 IC 扫描")
    parser.add_argument("--name", default="a158_csi500p_fin")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))
    ds = lab.load_dataset(args.name)
    if ds is None:
        print(f"ERROR: 找不到 dataset {args.name}")
        sys.exit(1)

    test_start, test_end = ds.data_periods[Segment.TEST]

    # 从 result_df 算 T+1 口径未来收益(按股票)
    res = ds.result_df.sort(["vt_symbol", "datetime"])
    base = pl.col("close").shift(-1).over("vt_symbol")          # t+1 开盘(用收盘近似)买入基准
    res = res.with_columns([
        (pl.col("close").shift(-2).over("vt_symbol") / base - 1).alias("fwd_1d"),
        (pl.col("close").shift(-3).over("vt_symbol") / base - 1).alias("fwd_2d"),
        (pl.col("close").shift(-6).over("vt_symbol") / base - 1).alias("fwd_5d"),
    ])
    fwd = res.select(["datetime", "vt_symbol", "fwd_1d", "fwd_2d", "fwd_5d"])

    # 因子来自 learn_df(含 158 量价 + 财务)
    learn = ds.learn_df
    factor_cols = [c for c in learn.columns if c not in ("datetime", "vt_symbol", "label")]

    merged = learn.join(fwd, on=["datetime", "vt_symbol"], how="inner").filter(
        (pl.col("datetime") >= pl.lit(test_start).str.to_datetime())
        & (pl.col("datetime") <= pl.lit(test_end).str.to_datetime())
    )
    print(f"OOS {test_start}~{test_end}, {len(merged):,} 行, {len(factor_cols)} 因子")

    rows = []
    for c in factor_cols:
        _, ir1 = daily_icir(merged, c, "fwd_1d")
        _, ir2 = daily_icir(merged, c, "fwd_2d")
        _, ir5 = daily_icir(merged, c, "fwd_5d")
        rows.append({"factor": c, "icir_1d": ir1, "icir_2d": ir2, "icir_5d": ir5})

    # 按 1 日 |ICIR| 排序(最短线)
    rows.sort(key=lambda r: -abs(r["icir_1d"]))

    print(f"\n=== 短线最强因子(按 1日 |ICIR| 排序, Top {args.top})===")
    print(f"{'因子':<14}{'1日ICIR':>9}{'2日ICIR':>9}{'5日ICIR':>9}  衰减趋势")
    print("-" * 56)
    for r in rows[:args.top]:
        trend = "短线特有" if abs(r["icir_1d"]) > abs(r["icir_5d"]) * 1.3 else (
            "中线更强" if abs(r["icir_5d"]) > abs(r["icir_1d"]) * 1.3 else "全周期")
        print(f"{r['factor']:<14}{r['icir_1d']:>+9.3f}{r['icir_2d']:>+9.3f}{r['icir_5d']:>+9.3f}  {trend}")

    # 短线特有因子(1日强、5日明显弱)
    shortonly = [r for r in rows if abs(r["icir_1d"]) > 0.15 and abs(r["icir_1d"]) > abs(r["icir_5d"]) * 1.5]
    print(f"\n=== 短线特有因子(1日 |ICIR|>0.15 且明显强于 5日)===")
    for r in sorted(shortonly, key=lambda r: -abs(r["icir_1d"]))[:15]:
        print(f"  {r['factor']:<14} 1日{r['icir_1d']:+.3f} vs 5日{r['icir_5d']:+.3f}")


if __name__ == "__main__":
    main()
