"""
当前市场有效因子分析(Phase 4)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_analyze.py --name a158_mktcap500

输出：
    1. LightGBM gain 因子重要性 Top-N
    2. OOS 各因子截面 Spearman IC / ICIR / IC>0%
    3. 因子类别 ICIR 汇总(动量/波动/量价相关/反转等)
    → 形成"当前市场有效因子组合"结论
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset.utility import Segment

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"

# 因子类别归类(按 Alpha158 命名前缀)
CATEGORY_RULES = {
    "量价相关(corr/cord)": lambda c: c.startswith("corr") or c.startswith("cord"),
    "成交量(v*)": lambda c: c.startswith("v") and not c.startswith("vwap"),
    "动量(roc)": lambda c: c.startswith("roc"),
    "波动率(std/beta)": lambda c: c.startswith("std") or c.startswith("beta"),
    "线性残差(resi/rsqr)": lambda c: c.startswith("resi") or c.startswith("rsqr"),
    "分位/位置(qtl/rsv/min/max)": lambda c: c.startswith(("qtl", "rsv", "min", "max")),
    "K线形态(k*)": lambda c: c.startswith("k"),
}


def feature_importance(model, top_n: int = 30) -> None:
    booster = model.model
    importance = booster.feature_importance(importance_type="gain")
    features = booster.feature_name()
    fi = sorted(zip(features, importance), key=lambda x: -x[1])
    total = sum(importance) or 1

    print(f"\n=== LightGBM Gain 重要性 Top {top_n} ===")
    cum = 0.0
    for i, (name, imp) in enumerate(fi[:top_n], 1):
        cum += imp
        print(f"{i:2d}. {name:<12} {imp/total*100:5.2f}%  累计{cum/total*100:5.1f}%")
    print(f"Top10 覆盖 {sum(v for _, v in fi[:10])/total*100:.1f}% | "
          f"Top20 覆盖 {sum(v for _, v in fi[:20])/total*100:.1f}%")


def oos_factor_ic(dataset) -> list[dict]:
    """OOS 期每个因子的截面 Spearman IC"""
    test_df = dataset.fetch_raw(Segment.TEST)
    factor_cols = [c for c in test_df.columns if c not in ("datetime", "vt_symbol", "label")]

    results: list[dict] = []
    for col in factor_cols:
        sub = test_df.select(["datetime", col, "label"]).drop_nulls()
        if len(sub) < 100:
            continue
        daily_ic: list[float] = []
        for _key, day in sub.group_by("datetime"):
            if len(day) < 10:
                continue
            x = day[col].to_numpy()
            y = day["label"].to_numpy()
            m = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if m.sum() < 10:
                continue
            ic, _ = spearmanr(x[m], y[m])
            if not np.isnan(ic):
                daily_ic.append(ic)
        if not daily_ic:
            continue
        arr = np.array(daily_ic)
        results.append({
            "factor": col,
            "IC_mean": arr.mean(),
            "IC_std": arr.std(),
            "ICIR": arr.mean() / (arr.std() + 1e-12),
            "IC_pos": (arr > 0).mean(),
        })

    results.sort(key=lambda r: -abs(r["ICIR"]))
    return results


def print_ic_table(results: list[dict], top_n: int = 25) -> None:
    print(f"\n=== OOS 因子 IC(按 |ICIR| 排序，Top {top_n}) ===")
    print(f"{'因子':<12} {'IC均值':>9} {'IC_std':>7} {'ICIR':>8} {'IC>0%':>7}")
    print("-" * 50)
    for r in results[:top_n]:
        print(f"{r['factor']:<12} {r['IC_mean']:>+9.4f} {r['IC_std']:>7.4f} "
              f"{r['ICIR']:>+8.3f} {r['IC_pos']:>6.1%}")


def print_category_summary(results: list[dict]) -> None:
    print("\n=== 因子类别 ICIR 汇总 ===")
    for cat, rule in CATEGORY_RULES.items():
        items = [r for r in results if rule(r["factor"])]
        if not items:
            continue
        avg_icir = np.mean([abs(r["ICIR"]) for r in items])
        avg_ic = np.mean([r["IC_mean"] for r in items])
        print(f"{cat:<26} n={len(items):2d}  avg|ICIR|={avg_icir:.3f}  avg_IC={avg_ic:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="当前市场有效因子分析")
    parser.add_argument("--name", default="a158_mktcap500", help="实验名称")
    parser.add_argument("--top-n", type=int, default=30, help="重要性/IC 展示条数")
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))
    model = lab.load_model(args.name)
    dataset = lab.load_dataset(args.name)
    if model is None or dataset is None:
        print(f"ERROR: 找不到 {args.name} 的 model/dataset")
        sys.exit(1)

    feature_importance(model, args.top_n)
    results = oos_factor_ic(dataset)
    print_ic_table(results, args.top_n)
    print_category_summary(results)


if __name__ == "__main__":
    main()
