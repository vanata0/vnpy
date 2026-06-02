"""
导出 point-in-time 财务因子(攻因子 — 成长/质量,量价无法隐含的正交信息)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/export_financials.py

核心：DuckDB ASOF JOIN 实现 point-in-time —— 每个交易日只匹配 notice_date(公告日)
<= 当日的最新财报，从根上杜绝未来函数(绝不用还没公布的财报)。

产出 lab_data/financials.parquet，列：
    datetime, vt_symbol,
    np_yoy(净利同比增速,成长) rev_yoy(营收同比增速,成长)
    roe(加权ROE,质量) gross_margin(毛利率,质量) net_margin(净利率,质量)
    debt_ratio(资产负债率,杠杆) ocf_ps(每股经营现金流,现金流质量)

与 export_fundamental.py(pe/pb/换手率/量比,粗估值)互补：这里是成长+质量维度。
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
START = "2017-09-01"

# ASOF JOIN：左表每个交易日，匹配该股 notice_date <= date 的最新财报
SQL = """
SELECT
    d.date::TIMESTAMP AS datetime,
    d.code,
    f.net_profit_yoy_gr AS np_yoy,
    f.total_rev_yoy_gr  AS rev_yoy,
    f.roe_wtd           AS roe,
    f.gross_margin      AS gross_margin,
    f.net_margin        AS net_margin,
    f.asset_liab_ratio  AS debt_ratio,
    f.oper_cf_ps        AS ocf_ps
FROM stock_daily d
ASOF LEFT JOIN financial_report f
    ON d.code = f.code AND d.date >= f.notice_date
WHERE d.date >= ?
ORDER BY d.date, d.code
"""


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
        t1 = time.monotonic()
        df: pl.DataFrame = con.execute(SQL, [START]).pl()
        print(f"ASOF JOIN 完成 {time.monotonic()-t1:.1f}s", flush=True)
    finally:
        con.close()
        tmp_db.unlink(missing_ok=True)

    df = df.with_columns(
        pl.col("code").map_elements(to_vt_symbol, return_dtype=pl.Utf8).alias("vt_symbol")
    ).drop("code")
    df = df.select(["datetime", "vt_symbol", "np_yoy", "rev_yoy", "roe",
                    "gross_margin", "net_margin", "debt_ratio", "ocf_ps"])

    out = LAB_PATH / "financials.parquet"
    df.write_parquet(str(out))

    matched = df.filter(pl.col("roe").is_not_null()).height
    print(f"财务因子: {len(df):,} 行 → {out}")
    print(f"覆盖 {df['datetime'].min().date()} ~ {df['datetime'].max().date()}, "
          f"{df['vt_symbol'].n_unique()} 只, ROE 非空 {matched/len(df)*100:.1f}%")


if __name__ == "__main__":
    main()
