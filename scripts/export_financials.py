"""
导出 point-in-time 财务因子(攻因子 — 成长/质量/SUE,量价无法隐含的正交信息)

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
    delta_roe(ROE同比变化,质量趋势) cfo_quality(OCF/EPS应计质量) rev_accel(营收增速加速度)
    sue(EPS标准化未预期盈余) rev_sue(营收标准化未预期盈余)

与 export_fundamental.py(pe/pb/换手率/量比,粗估值)互补：这里是成长+质量+surprise维度。
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

# 两个 ASOF JOIN：
#   enriched_reports — 质量/成长/趋势因子（按 notice_date 匹配最新财报）
#   sue_q            — SUE surprise 因子（同 notice_date 时序，季报驱动）
#
# SUE 算法：
#   1. 原始季报 basic_eps / total_rev 是累积 YTD → 差分转单季纯值
#   2. YoY diff = 本季纯值 - 4季前纯值（同季度对比，去除季节性）
#   3. SUE = YoY_diff / rolling_std(8季度窗口)（标准化，跨股可比）
SQL = """
WITH enriched_reports AS (
    SELECT
        code, report_date, notice_date,
        net_profit_yoy_gr,
        total_rev_yoy_gr,
        roe_wtd,
        gross_margin,
        net_margin,
        asset_liab_ratio,
        oper_cf_ps,
        roe_wtd - LAG(roe_wtd, 1) OVER (
            PARTITION BY code, EXTRACT(MONTH FROM report_date)
            ORDER BY report_date
        ) AS delta_roe,
        CASE WHEN ABS(basic_eps) >= 0.01
             THEN oper_cf_ps / basic_eps
             ELSE NULL END AS cfo_quality,
        total_rev_yoy_gr - LAG(total_rev_yoy_gr, 1) OVER (
            PARTITION BY code, EXTRACT(MONTH FROM report_date)
            ORDER BY report_date
        ) AS rev_accel,
    FROM financial_report
),
-- 取季报（3/6/9/12月末）的原始累积 YTD 数据
raw_q AS (
    SELECT code, report_date, notice_date, basic_eps, total_rev,
           EXTRACT(YEAR  FROM report_date) AS yr,
           EXTRACT(MONTH FROM report_date) AS qm
    FROM financial_report
    WHERE EXTRACT(MONTH FROM report_date) IN (3, 6, 9, 12)
      AND basic_eps IS NOT NULL AND total_rev IS NOT NULL
),
-- 累积 YTD → 单季纯值（Q1 即 YTD；Q2-Q4 减去本年上期累积）
pure_q AS (
    SELECT code, report_date, notice_date,
           CASE WHEN qm = 3 THEN basic_eps
                ELSE basic_eps - LAG(basic_eps) OVER (PARTITION BY code, yr ORDER BY report_date)
           END AS eps_q,
           CASE WHEN qm = 3 THEN total_rev
                ELSE total_rev - LAG(total_rev) OVER (PARTITION BY code, yr ORDER BY report_date)
           END AS rev_q
    FROM raw_q
),
-- 同季度 YoY 差值（本季 - 4期前同季）
yoy_q AS (
    SELECT code, report_date, notice_date,
           eps_q - LAG(eps_q, 4) OVER (PARTITION BY code ORDER BY report_date) AS eps_diff,
           rev_q - LAG(rev_q, 4) OVER (PARTITION BY code ORDER BY report_date) AS rev_diff
    FROM pure_q
    WHERE eps_q IS NOT NULL AND rev_q IS NOT NULL
),
-- SUE = YoY_diff / rolling 8季度标准差（标准化，需至少 4 期有效数据）
sue_q AS (
    SELECT code, notice_date,
           eps_diff / NULLIF(STDDEV(eps_diff) OVER w, 0) AS sue,
           rev_diff / NULLIF(STDDEV(rev_diff) OVER w, 0) AS rev_sue
    FROM yoy_q
    WHERE eps_diff IS NOT NULL AND rev_diff IS NOT NULL
    WINDOW w AS (PARTITION BY code ORDER BY report_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW)
)
SELECT
    d.date::TIMESTAMP AS datetime,
    d.code,
    f.net_profit_yoy_gr AS np_yoy,
    f.total_rev_yoy_gr  AS rev_yoy,
    f.roe_wtd           AS roe,
    f.gross_margin      AS gross_margin,
    f.net_margin        AS net_margin,
    f.asset_liab_ratio  AS debt_ratio,
    f.oper_cf_ps        AS ocf_ps,
    f.delta_roe         AS delta_roe,
    f.cfo_quality       AS cfo_quality,
    f.rev_accel         AS rev_accel,
    s.sue               AS sue,
    s.rev_sue           AS rev_sue
FROM stock_daily d
ASOF LEFT JOIN enriched_reports f
    ON d.code = f.code AND d.date >= f.notice_date
ASOF LEFT JOIN sue_q s
    ON d.code = s.code AND d.date >= s.notice_date
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
                    "gross_margin", "net_margin", "debt_ratio", "ocf_ps",
                    "delta_roe", "cfo_quality", "rev_accel", "sue", "rev_sue"])

    out = LAB_PATH / "financials.parquet"
    df.write_parquet(str(out))

    matched = df.filter(pl.col("roe").is_not_null()).height
    sue_matched = df.filter(pl.col("sue").is_not_null()).height
    print(f"财务因子: {len(df):,} 行 → {out}")
    print(f"覆盖 {df['datetime'].min().date()} ~ {df['datetime'].max().date()}, "
          f"{df['vt_symbol'].n_unique()} 只, ROE 非空 {matched/len(df)*100:.1f}%")
    print(f"SUE 非空 {sue_matched/len(df)*100:.1f}%  "
          f"| SUE p25={df['sue'].quantile(0.25):.2f} p50={df['sue'].quantile(0.5):.2f} "
          f"p75={df['sue'].quantile(0.75):.2f}")


if __name__ == "__main__":
    main()
