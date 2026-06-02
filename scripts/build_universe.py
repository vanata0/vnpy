"""
构建动态选股 universe + benchmark 导入(Phase 1)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/build_universe.py                 # 默认 Top500 月度
    python scripts/build_universe.py --top-n 300

产出：
    lab_data/component/MKTCAP500*       AlphaLab 虚拟指数成分股(月度 Top-N)
    lab_data/limit_status.parquet       每日涨跌停标记(仅涨跌停行)
    lab_data/daily/000300.SSE.parquet   沪深300 指数日线(回测 benchmark)

设计说明：
    - 排名用 mktcap(总市值)。stock_new 的 float_cap 全表缺失，mktcap 100% 覆盖。
    - 每月末交易日取市值 Top-N，过滤 ST 与上市不足 N 天的新股。
    - 存成 AlphaLab component 格式，复用 load_component_symbols/filters 接入训练与回测。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab

SRC_DB = "/home/oracle/stock_new/data/db/market.duckdb"
LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"

# universe 名单从 2017-12 月末开始，使 IS 2018-01 起的数据有成分覆盖
UNIVERSE_START = "2017-12-01"
INDEX_SYMBOL = "MKTCAP500"


def to_vt_symbol(code: str) -> str:
    """与 alpha158_bridge.py 保持一致的 vt_symbol 转换"""
    if code.startswith(("8", "9")):
        return f"{code}.BSE"
    if code.startswith("6"):
        return f"{code}.SSE"
    return f"{code}.SZSE"


def clean_component(lab: AlphaLab) -> None:
    """清理旧的 MKTCAP500 shelve 文件，避免重跑残留"""
    for f in lab.component_path.glob(f"{INDEX_SYMBOL}*"):
        f.unlink()


def build_universe(con: duckdb.DuckDBPyConnection, lab: AlphaLab, top_n: int, min_list_days: int) -> None:
    """生成月度 Top-N 市值成分股"""
    # 取每月末交易日
    month_ends: list = (
        con.execute(
            """
            SELECT MAX(date) AS month_end
            FROM stock_daily
            WHERE date >= ?
            GROUP BY date_trunc('month', date)
            ORDER BY month_end
            """,
            [UNIVERSE_START],
        )
        .fetchdf()["month_end"]
        .tolist()
    )

    components: dict[str, list[str]] = {}
    for me in month_ends:
        me_str = me.strftime("%Y-%m-%d")
        codes: list[str] = (
            con.execute(
                """
                SELECT d.code
                FROM stock_daily d
                JOIN stock_meta m ON d.code = m.code
                WHERE d.date = ?
                  AND d.mktcap IS NOT NULL
                  AND (d.is_st = false OR d.is_st IS NULL)
                  AND m.list_date <= (CAST(? AS DATE) - INTERVAL '%d' DAY)
                ORDER BY d.mktcap DESC
                LIMIT ?
                """ % min_list_days,
                [me, me, top_n],
            )
            .fetchdf()["code"]
            .tolist()
        )
        components[me_str] = [to_vt_symbol(c) for c in codes]

    clean_component(lab)
    lab.save_component_data(INDEX_SYMBOL, components)

    sizes = [len(v) for v in components.values()]
    print(f"universe: {len(components)} 个月度快照, 每月 {min(sizes)}~{max(sizes)} 只 "
          f"({month_ends[0].strftime('%Y-%m')} ~ {month_ends[-1].strftime('%Y-%m')})")


def export_limit_status(con: duckdb.DuckDBPyConnection) -> None:
    """导出每日涨跌停标记(仅存涨停或跌停的行，省空间)"""
    df: pl.DataFrame = con.execute(
        """
        SELECT
            date::TIMESTAMP AS datetime,
            code,
            is_limitup,
            is_limitdown
        FROM stock_daily
        WHERE (is_limitup OR is_limitdown) AND date >= ?
        ORDER BY date, code
        """,
        [UNIVERSE_START],
    ).pl()

    df = df.with_columns(
        pl.col("code").map_elements(to_vt_symbol, return_dtype=pl.Utf8).alias("vt_symbol")
    ).select(["datetime", "vt_symbol", "is_limitup", "is_limitdown"])

    out = LAB_PATH / "limit_status.parquet"
    df.write_parquet(str(out))
    print(f"limit_status: {len(df):,} 行涨跌停记录 → {out}")


def export_benchmark(con: duckdb.DuckDBPyConnection) -> None:
    """导出沪深300 指数日线作 benchmark(index_daily 仅 close 可用，OHL 用 close 填充)"""
    df: pl.DataFrame = con.execute(
        """
        SELECT
            date::TIMESTAMP AS datetime,
            close AS open,
            close AS high,
            close AS low,
            close,
            0 AS volume,
            0.0 AS turnover,
            0 AS open_interest
        FROM index_daily
        WHERE code = '000300.SH' AND close IS NOT NULL
        ORDER BY date
        """
    ).pl()

    out = LAB_PATH / "daily" / "000300.SSE.parquet"
    df.write_parquet(str(out))
    print(f"benchmark 000300.SSE: {len(df):,} 行 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建选股 universe + benchmark")
    parser.add_argument("--top-n", type=int, default=500, help="每月市值 Top-N")
    parser.add_argument("--min-list-days", type=int, default=60, help="上市最少天数(剔新股)")
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))

    fd, tmp_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    tmp_db = Path(tmp_path)
    print(f"复制 DB → {tmp_db}", flush=True)
    t0 = time.monotonic()
    shutil.copy2(SRC_DB, tmp_db)
    print(f"复制完成 {time.monotonic()-t0:.1f}s", flush=True)

    try:
        con = duckdb.connect(str(tmp_db), read_only=False)
        build_universe(con, lab, args.top_n, args.min_list_days)
        export_limit_status(con)
        export_benchmark(con)
    finally:
        con.close()
        tmp_db.unlink(missing_ok=True)

    print("\nPhase 1 完成")


if __name__ == "__main__":
    main()
