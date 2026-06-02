"""
Alpha158 数据桥：stock_new market.duckdb → AlphaLab daily parquet

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_bridge.py              # 全量
    python scripts/alpha158_bridge.py --test 5     # 只处理前 N 只（测试用）
    python scripts/alpha158_bridge.py --time 100   # 跑 100 只并报告耗时
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import polars as pl

SRC_DB = "/home/oracle/stock_new/data/db/market.duckdb"
OUT_DIR = Path("/home/oracle/vnpy/lab_data/daily")

SQL = """
SELECT
    date::TIMESTAMP AS datetime,
    open_qfq  AS open,
    high_qfq  AS high,
    low_qfq   AS low,
    close_qfq AS close,
    volume,
    -- turnover 存复权成交额，使 load_bar_df 的 turnover/volume = vwap_qfq
    CASE
        WHEN close > 0 THEN amount * (close_qfq / close)
        ELSE amount
    END AS turnover,
    -- 停牌日 vwap 用 close_qfq 占位；load_bar_df 重算时停牌日结果为 NaN
    CASE
        WHEN volume > 0 AND close > 0 THEN (amount / volume) * (close_qfq / close)
        ELSE close_qfq
    END AS vwap,
    0 AS open_interest
FROM stock_daily
WHERE code = ?
ORDER BY date
"""


def to_vt_symbol(code: str) -> str:
    if code.startswith(("8", "9")):
        return f"{code}.BSE"
    if code.startswith("6"):
        return f"{code}.SSE"
    return f"{code}.SZSE"


def run(limit: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    tmp_db = Path(tmp_path)

    print(f"复制 DB: {SRC_DB} → {tmp_db}", flush=True)
    t0 = time.monotonic()
    shutil.copy2(SRC_DB, tmp_db)
    print(f"复制完成，耗时 {time.monotonic() - t0:.1f}s", flush=True)

    try:
        con = duckdb.connect(str(tmp_db), read_only=False)
        codes: list[str] = (
            con.execute("SELECT DISTINCT code FROM stock_daily ORDER BY code")
            .fetchdf()["code"]
            .tolist()
        )

        if limit is not None:
            codes = codes[:limit]

        total = len(codes)
        print(f"开始处理 {total} 只股票 → {OUT_DIR}", flush=True)

        t_start = time.monotonic()
        for i, code in enumerate(codes, 1):
            vt_symbol = to_vt_symbol(code)
            df: pl.DataFrame = con.execute(SQL, [code]).pl()

            out_path = OUT_DIR / f"{vt_symbol}.parquet"
            tmp_parquet = out_path.with_suffix(".parquet.tmp")
            df.write_parquet(str(tmp_parquet))
            tmp_parquet.replace(out_path)

            if i % 500 == 0 or i == total:
                elapsed = time.monotonic() - t_start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{total}] {elapsed:.0f}s 已用 | "
                    f"{rate:.0f} 只/s | ETA {eta:.0f}s",
                    flush=True,
                )

    finally:
        con.close()
        tmp_db.unlink(missing_ok=True)

    elapsed = time.monotonic() - t_start
    print(f"\n完成：{total} 只，{elapsed:.1f}s（{total/elapsed:.0f} 只/s）")
    print(f"输出目录：{OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha158 数据桥")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test", type=int, metavar="N", help="只处理前 N 只股票")
    group.add_argument("--time", type=int, metavar="N", help="跑 N 只并报告耗时（用于估算全量时间）")
    args = parser.parse_args()

    limit = args.test or args.time
    run(limit=limit)

    if args.time:
        parquet_count = len(list(OUT_DIR.glob("*.parquet")))
        print(f"\n计时模式：已写出 {parquet_count} 个 parquet 文件")


if __name__ == "__main__":
    main()
