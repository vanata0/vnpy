"""
一键全流程编排：数据更新 → 增量推理 → 回测 → 出信号

两种模式：
    日常更新（默认）— 增量推理，~5 分钟
        python scripts/run_pipeline.py

    首次建库 / 季度重训 — 全量重建，~40 分钟
        python scripts/run_pipeline.py --full

    从第 N 步继续（两种模式通用）：
        python scripts/run_pipeline.py --from 3

日常模式：增量脚本自动检测新交易日，无需手动更新 END_DATE。
全量模式：train.py 的 TEST_PERIOD、rolling.py 的 W4 test、本文件 FULL_END_DATE 需同步更新。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/oracle/vnpy-venv/bin/python"

# 全量模式回测终止日（仅当 signal 尚未生成时的兜底值）
FULL_END_DATE = "2026-06-04"
_END = "@END"  # 运行时替换为 signal 最新日期的占位符

STEPS_INCREMENTAL: list[tuple[str, list[str]]] = [
    ("指数 benchmark + 官方成分股", ["scripts/fetch_index_tushare.py"]),
    ("财务因子 point-in-time",      ["scripts/export_financials.py"]),
    ("行业分类导出",                 ["scripts/export_industry.py"]),
    ("增量推理 + 选股清单 (~5min)",  ["scripts/alpha158_infer_incremental.py"]),
    ("回测 + 净值曲线",              ["scripts/alpha158_backtest.py",
                                      "--name", "a158_csi500_fin_rolling",
                                      "--start", "2023-01-01", "--end", _END,
                                      "--t1", "--filter-limit", "--industry-neutral"]),
]

STEPS_FULL: list[tuple[str, list[str]]] = [
    ("Tushare 日线全量重载",         ["scripts/fetch_daily_tushare.py", "--full"]),
    ("指数 benchmark + 官方成分股", ["scripts/fetch_index_tushare.py"]),
    ("财务因子 point-in-time",      ["scripts/export_financials.py"]),
    ("行业分类导出",                 ["scripts/export_industry.py"]),
    ("训练 LightGBM (~35min)",      ["scripts/alpha158_train.py",
                                      "--name", "a158_csi500_fin",
                                      "--universe", "CSI500",
                                      "--batch-size", "100", "--workers", "4",
                                      "--financial"]),
    ("滚动 OOS 验证",                ["scripts/alpha158_rolling.py",
                                      "--base", "a158_csi500_fin"]),
    ("回测 + 净值曲线",              ["scripts/alpha158_backtest.py",
                                      "--name", "a158_csi500_fin_rolling",
                                      "--start", "2023-01-01", "--end", _END,
                                      "--t1", "--filter-limit", "--industry-neutral"]),
    ("生成最新选股清单",             ["scripts/generate_signal.py", "--top-k", "50"]),
]


def detect_signal_end(signal_name: str = "a158_csi500_fin_rolling") -> str:
    """读取已保存 rolling signal 的最新交易日，失败时降级为 FULL_END_DATE。"""
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from vnpy.alpha import AlphaLab
        sig = AlphaLab(str(ROOT / "lab_data")).load_signal(signal_name)
        if sig is not None:
            return str(sig["datetime"].max().date())
    except Exception:
        pass
    return FULL_END_DATE


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha158 全流程编排")
    ap.add_argument("--full", action="store_true", help="全量重建模式（首次建库/季度重训）")
    ap.add_argument("--from", dest="start", type=int, default=1, help="从第 N 步开始(1-based)")
    args = ap.parse_args()

    steps = STEPS_FULL if args.full else STEPS_INCREMENTAL
    mode = "全量重建" if args.full else "增量推理"

    print(f"模式: {mode}  步骤数: {len(steps)}")

    t0 = time.monotonic()
    end_date: str | None = None  # 延迟到回测步骤前才检测（先让推理步骤更新 signal）

    for i, (name, cmd) in enumerate(steps, 1):
        if i < args.start:
            print(f"[{i}/{len(steps)}] {name} — 跳过", flush=True)
            continue

        # 遇到 @END 占位符时检测最新 signal 日期（推理步骤已完成）
        if _END in cmd:
            if end_date is None:
                end_date = detect_signal_end()
                print(f"回测终止日自动检测: {end_date}", flush=True)
            cmd = [end_date if c == _END else c for c in cmd]

        print(f"\n{'='*56}\n[{i}/{len(steps)}] {name}\n{'='*56}", flush=True)
        ts = time.monotonic()
        r = subprocess.run([PY, *cmd], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n❌ [{name}] 失败(退出码 {r.returncode}),流程中止", flush=True)
            sys.exit(1)
        print(f"✓ {name} — {time.monotonic()-ts:.0f}s", flush=True)

    elapsed = (time.monotonic() - t0) / 60
    print(f"\n✓ {mode} 完成，总耗时 {elapsed:.1f} 分钟")


if __name__ == "__main__":
    main()
