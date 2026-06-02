"""
Alpha158 LightGBM 训练流程

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_train.py              # 全量（~5770只，耗时较长）
    python scripts/alpha158_train.py --limit 200  # 只用前 N 只股票（测试用）
    python scripts/alpha158_train.py --exchange SSE --limit 500

训练区间配置（在脚本底部 main() 里修改）：
    IS:    2015-01-01 ~ 2021-12-31
    Valid: 2022-01-01 ~ 2022-12-31
    OOS:   2023-01-01 ~ 2024-12-31
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import process_cs_norm, process_drop_na
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.utility import Segment
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.trader.constant import Interval

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
DAILY_PATH = LAB_PATH / "daily"


def get_symbols(exchange: str | None = None, limit: int | None = None) -> list[str]:
    """从 lab_data/daily 扫描可用的 vt_symbol 列表"""
    parquets = sorted(DAILY_PATH.glob("*.parquet"))
    symbols = [p.stem for p in parquets]

    if exchange:
        symbols = [s for s in symbols if s.endswith(f".{exchange.upper()}")]

    if limit:
        symbols = symbols[:limit]

    return symbols


def run(
    name: str,
    symbols: list[str],
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_period: tuple[str, str],
    extended_days: int = 100,
    max_workers: int = 8,
) -> None:
    lab = AlphaLab(str(LAB_PATH))

    print(f"股票池: {len(symbols)} 只")
    print(f"IS:    {train_period[0]} ~ {train_period[1]}")
    print(f"Valid: {valid_period[0]} ~ {valid_period[1]}")
    print(f"OOS:   {test_period[0]} ~ {test_period[1]}")

    # 加载 bar 数据
    t0 = time.monotonic()
    start = train_period[0]
    end = test_period[1]
    df = lab.load_bar_df(symbols, Interval.DAILY, start, end, extended_days)
    if df is None or df.is_empty():
        print("ERROR: load_bar_df 返回空，检查 lab_data/daily/ 是否有数据")
        sys.exit(1)
    print(f"加载完成: {df.shape[0]:,} 行，耗时 {time.monotonic()-t0:.1f}s")

    # 构建 Alpha158 数据集
    dataset = Alpha158(df, train_period=train_period, valid_period=valid_period, test_period=test_period)

    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    # 计算 158 个因子
    t1 = time.monotonic()
    dataset.prepare_data(filters=None, max_workers=max_workers)
    print(f"因子计算完成，耗时 {time.monotonic()-t1:.1f}s")

    dataset.process_data()
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
    import numpy as np
    print("\n--- 测试集信号绩效 ---")
    pre = model.predict(dataset, Segment.TEST)
    df_t = dataset.fetch_infer(Segment.TEST)
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal = df_t["datetime", "vt_symbol", "signal"]

    lab.save_signal(name, signal)
    dataset.show_signal_performance(signal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha158 LightGBM 训练")
    parser.add_argument("--name", default="a158_lgb", help="实验名称（用于保存 dataset/model/signal）")
    parser.add_argument("--exchange", choices=["SSE", "SZSE", "BSE"], default=None, help="只用某交易所的股票")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量（测试用）")
    parser.add_argument("--workers", type=int, default=8, help="并行因子计算线程数")
    args = parser.parse_args()

    symbols = get_symbols(exchange=args.exchange, limit=args.limit)
    if not symbols:
        print(f"ERROR: 没有找到符合条件的 parquet 文件")
        sys.exit(1)

    run(
        name=args.name,
        symbols=symbols,
        train_period=("2015-01-01", "2021-12-31"),
        valid_period=("2022-01-01", "2022-12-31"),
        test_period=("2023-01-01", "2024-12-31"),
        extended_days=100,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
