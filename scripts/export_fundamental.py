"""
导出基本面因子数据(攻因子 — 正交于纯量价的信息源)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/export_fundamental.py

产出 lab_data/fundamental.parquet，列：
    datetime, vt_symbol, pe, pb, turnover_rate, vol_ratio

字段说明(均来自 stock_daily，每日现成、无财报未来函数)：
    pe            市盈率(估值，可为负=亏损)
    pb            市净率(估值)
    turnover_rate 换手率%(流动性/情绪)；注意与 daily parquet 的 turnover(成交额)不同
    vol_ratio     量比(量能)

原始值保留(含负 pe 等极端值)，截面 rank 标准化在 train 阶段做。
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC_DB = "/home/oracle/stock_new/data/db/market.duckdb"
LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
START = "2017-09-01"  # 早于 universe 起点，留因子预热余量


def to_vt_symbol(code: str) -> str:
    if code.startswith(("8", "9")):
        return f"{code}.BSE"
    if code.startswith("6"):
        return f"{code}.SSE"
    return f"{code}.SZSE"


def main() -> None:
    fd, tmp_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    tmp_db = Path(tmp_path)
    print(f"复制 DB → {tmp_db}", flush=True)
    t0 = time.monotonic()
    shutil.copy2(SRC_DB, tmp_db)
    print(f"复制完成 {time.monotonic()-t0:.1f}s", flush=True)

    try:
        con = duckdb.connect(str(tmp_db), read_only=False)
        df: pl.DataFrame = con.execute(
            """
            SELECT
                date::TIMESTAMP AS datetime,
                code,
                pe,
                pb,
                turnover AS turnover_rate,
                vol_ratio
            FROM stock_daily
            WHERE date >= ?
            ORDER BY date, code
            """,
            [START],
        ).pl()
    finally:
        con.close()
        tmp_db.unlink(missing_ok=True)

    df = df.with_columns(
        pl.col("code").map_elements(to_vt_symbol, return_dtype=pl.Utf8).alias("vt_symbol")
    ).select(["datetime", "vt_symbol", "pe", "pb", "turnover_rate", "vol_ratio"])

    out = LAB_PATH / "fundamental.parquet"
    df.write_parquet(str(out))
    print(f"基本面: {len(df):,} 行 → {out}")
    print(f"覆盖 {df['datetime'].min().date()} ~ {df['datetime'].max().date()}, "
          f"{df['vt_symbol'].n_unique()} 只股票")


if __name__ == "__main__":
    main()
