"""
一键全流程编排：数据更新 → universe → 因子 → 训练 → 滚动 → 回测 → 出信号

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/run_pipeline.py            # 全流程(~30分钟)
    python scripts/run_pipeline.py --from 5   # 从第5步(训练)开始,跳过数据/因子重建

更新最新数据时：bridge/universe/financials 自动含最新交易日;但 train.py 的
TEST_PERIOD、rolling.py 的 W4 test、本文件 END_DATE 三处需同步改到最新交易日。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/oracle/vnpy-venv/bin/python"
END_DATE = "2026-06-04"  # 最新交易日;与 train.py TEST_PERIOD / rolling.py W4 保持一致

STEPS: list[tuple[str, list[str]]] = [
    ("数据桥 DuckDB→parquet", ["scripts/alpha158_bridge.py"]),
    ("指数 benchmark + 官方成分股", ["scripts/fetch_index_tushare.py"]),
    ("财务因子 point-in-time", ["scripts/export_financials.py"]),
    ("行业分类导出", ["scripts/export_industry.py"]),
    ("训练 LightGBM (~35min)", ["scripts/alpha158_train.py", "--name", "a158_csi500_fin", "--universe", "CSI500", "--batch-size", "100", "--workers", "4", "--financial"]),
    ("滚动 OOS 验证", ["scripts/alpha158_rolling.py", "--base", "a158_csi500_fin"]),
    ("回测 + 净值曲线", ["scripts/alpha158_backtest.py", "--name", "a158_csi500_fin_rolling", "--start", "2023-01-01", "--end", END_DATE, "--t1", "--filter-limit", "--industry-neutral"]),
    ("生成最新选股清单", ["scripts/generate_signal.py", "--top-k", "50"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha158 全流程编排")
    ap.add_argument("--from", dest="start", type=int, default=1, help="从第 N 步开始(1-based)")
    args = ap.parse_args()

    t0 = time.monotonic()
    for i, (name, cmd) in enumerate(STEPS, 1):
        if i < args.start:
            print(f"[{i}/{len(STEPS)}] {name} — 跳过", flush=True)
            continue
        print(f"\n{'='*56}\n[{i}/{len(STEPS)}] {name}\n{'='*56}", flush=True)
        ts = time.monotonic()
        r = subprocess.run([PY, *cmd], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n❌ [{name}] 失败(退出码 {r.returncode}),流程中止", flush=True)
            sys.exit(1)
        print(f"✓ {name} — {time.monotonic()-ts:.0f}s", flush=True)

    print(f"\n🎉 全流程完成,总耗时 {(time.monotonic()-t0)/60:.1f} 分钟")
    print(f"   最新清单: lab_data/picks_{END_DATE.replace('-', '')}.csv")


if __name__ == "__main__":
    main()
