"""
滚动 OOS 验证(walk-forward)— 解决单一 17 个月样本太短

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_rolling.py --base a158_mktcap500

原理：因子矩阵与 train/test 切分无关，复用已训练 dataset 的全量因子矩阵
(2018-2026)，对每个滚动窗口仅重训 LightGBM(秒级)+ predict，无需重算因子。

窗口(anchored，训练窗扩展，贴近实盘定期重训)：
    W1: train 2018-2021 | valid 2022 | OOS 2023
    W2: train 2018-2022 | valid 2023 | OOS 2024
    W3: train 2018-2023 | valid 2024 | OOS 2025
    W4: train 2018-2024 | valid 2025 | OOS 2026-01~05

输出：各窗口 OOS IC(年度稳定性) + 拼接 OOS(2023-2026)整体 IC，
拼接 signal 存为 <base>_rolling，供 T+1 回测覆盖 3.5 年。
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
from vnpy.alpha.model.models.lgb_model import LgbModel

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"

WINDOWS = [
    {"name": "W1", "train": ("2018-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2", "train": ("2018-01-01", "2022-12-31"), "valid": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W3", "train": ("2018-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    {"name": "W4", "train": ("2018-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-12-31"), "test": ("2026-01-01", "2026-06-04")},
]


def build_windows(w4_end: str | None = None) -> list[dict]:
    ws = [w.copy() for w in WINDOWS]
    if w4_end:
        ws[-1] = {**ws[-1], "test": (ws[-1]["test"][0], w4_end)}
    return ws


def daily_rank_ic(signal: pl.DataFrame, learn_test: pl.DataFrame) -> dict:
    """计算 signal 对 label 的截面 rank IC"""
    merged = signal.join(
        learn_test.select(["datetime", "vt_symbol", "label"]),
        on=["datetime", "vt_symbol"], how="inner"
    ).drop_nulls()

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
        return {"ic": float("nan"), "icir": float("nan"), "pos": float("nan"), "days": 0}
    arr = np.array(ics)
    return {
        "ic": arr.mean(),
        "icir": arr.mean() / (arr.std() + 1e-12),
        "pos": (arr > 0).mean(),
        "days": len(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="滚动 OOS 验证")
    parser.add_argument("--base", default="a158_mktcap500", help="已训练的全量 dataset 名")
    parser.add_argument("--w4-end", default=None, help="覆盖 W4 测试期终止日(增量推理用，格式 YYYY-MM-DD)")
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))
    dataset = lab.load_dataset(args.base)
    if dataset is None:
        print(f"ERROR: 找不到 dataset {args.base}")
        sys.exit(1)

    windows = build_windows(args.w4_end)

    rows: list[dict] = []
    oos_signals: list[pl.DataFrame] = []

    for w in windows:
        # 改时间切分(因子矩阵复用)，重训
        dataset.data_periods = {
            Segment.TRAIN: w["train"],
            Segment.VALID: w["valid"],
            Segment.TEST: w["test"],
        }
        model = LgbModel(seed=42, log_evaluation_period=500)
        model.fit(dataset)

        pre = model.predict(dataset, Segment.TEST)
        infer_test = dataset.fetch_infer(Segment.TEST)
        sig = infer_test.with_columns(pl.Series(pre).alias("signal"))[["datetime", "vt_symbol", "signal"]]
        oos_signals.append(sig)

        learn_test = dataset.fetch_learn(Segment.TEST)
        ic = daily_rank_ic(sig, learn_test)
        rows.append({"name": w["name"], "train": f"{w['train'][0][:4]}-{w['train'][1][:4]}",
                     "oos": w["test"][0][:7], **ic})
        print(f"{w['name']} 训练{w['train'][0][:4]}-{w['train'][1][:4]} OOS{w['test'][0][:7]}: "
              f"IC={ic['ic']:+.4f} ICIR={ic['icir']:+.3f} IC>0={ic['pos']:.1%} 天数={ic['days']}",
              flush=True)

    # 拼接 OOS signal 存盘
    merged = pl.concat(oos_signals).sort(["datetime", "vt_symbol"])
    rolling_name = f"{args.base}_rolling"
    lab.save_signal(rolling_name, merged)

    # 拼接整体 IC：用每个窗口的 learn_test 拼接 label
    print("\n" + "=" * 60)
    print("=== 滚动 OOS 汇总 ===")
    print(f"{'窗口':<5}{'训练':<12}{'OOS':<10}{'IC':>9}{'ICIR':>8}{'IC>0%':>8}{'天数':>6}")
    print("-" * 60)
    for r in rows:
        print(f"{r['name']:<5}{r['train']:<12}{r['oos']:<10}{r['ic']:>+9.4f}{r['icir']:>+8.3f}"
              f"{r['pos']:>7.1%}{r['days']:>6}")

    ics = [r["ic"] for r in rows if not np.isnan(r["ic"])]
    print("-" * 60)
    print(f"年度 IC 均值={np.mean(ics):+.4f}  标准差={np.std(ics):.4f}  "
          f"全部为正={all(i > 0 for i in ics)}")
    print(f"\n拼接 OOS signal ({len(merged):,} 行, "
          f"{merged['datetime'].min().date()}~{merged['datetime'].max().date()}) "
          f"已存为: {rolling_name}")
    print(f"\n下一步 T+1 回测(3.5年):")
    print(f"  python scripts/alpha158_backtest.py --name {rolling_name} "
          f"--start 2023-01-01 --end 2026-05-29 --t1 --filter-limit")


if __name__ == "__main__":
    main()
