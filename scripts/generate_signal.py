"""
生成最新交易日选股清单(研究 → 应用)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/generate_signal.py                       # 默认最优模型，top50
    python scripts/generate_signal.py --top-k 30 --date 2026-05-29

逻辑：取模型对最新交易日(T 日)收盘后的 signal，按信号值排序选 top_k，
作为 T+1(次日)开盘买入清单。剔除 T 日涨停股(次日大概率高开难买)。

⚠️ 实盘提示：当前模型用截至训练期的数据，实盘应定期(如每月)用最新数据重训。
信号仅为研究输出，非投资建议。
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
SRC_DB = "/home/oracle/stock_new/data/db/market.duckdb"


def load_names() -> dict[str, str]:
    """从 stock_meta 取 code → name(用于清单可读)"""
    fd, tmp = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    shutil.copy2(SRC_DB, tmp)
    try:
        con = duckdb.connect(tmp, read_only=False)
        rows = con.execute("SELECT code, name FROM stock_meta").fetchdf()
    finally:
        con.close()
        os.unlink(tmp)
    return dict(zip(rows["code"], rows["name"]))


def code_of(vt_symbol: str) -> str:
    return vt_symbol.rsplit(".", 1)[0]


def apply_industry_neutral(signal: pl.DataFrame) -> pl.DataFrame:
    """截面行业中性化：信号减去所在申万 L1 行业的截面均值"""
    ind_path = LAB_PATH / "industry.parquet"
    if not ind_path.exists():
        print("⚠️  industry.parquet 不存在，跳过行业中性化")
        return signal
    industry = pl.read_parquet(str(ind_path)).select(["vt_symbol", "sw_l1_name"])
    sig = signal.join(industry, on="vt_symbol", how="left")
    sig = sig.with_columns(pl.col("sw_l1_name").fill_null("其他"))
    sig = sig.with_columns(
        (pl.col("signal") - pl.col("signal").mean().over(["datetime", "sw_l1_name"])).alias("signal")
    )
    return sig.select(["datetime", "vt_symbol", "signal"])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成最新选股清单")
    parser.add_argument("--name", default="a158_csi500_fin_rolling", help="信号名称(最优:官方CSI500财务因子滚动版)")
    parser.add_argument("--top-k", type=int, default=50, help="选股数量")
    parser.add_argument("--date", default=None, help="指定交易日(YYYY-MM-DD)，默认最新")
    parser.add_argument("--no-industry-neutral", action="store_true", help="关闭行业中性化(调试用)")
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))
    signal = lab.load_signal(args.name)
    if signal is None:
        print(f"ERROR: 找不到信号 {args.name}")
        sys.exit(1)

    # 行业中性化(与回测口径一致)
    if not args.no_industry_neutral:
        signal = apply_industry_neutral(signal)

    # 确定交易日
    if args.date:
        target = pl.Series([args.date]).str.to_datetime()[0]
    else:
        target = signal["datetime"].max()
    day = signal.filter(pl.col("datetime") == target)
    if day.is_empty():
        print(f"ERROR: {target} 无信号")
        sys.exit(1)

    # 剔除当日涨停(次日难买)
    limit = pl.read_parquet(LAB_PATH / "limit_status.parquet")
    up = limit.filter((pl.col("datetime") == target) & pl.col("is_limitup"))["vt_symbol"].to_list()
    day = day.filter(~pl.col("vt_symbol").is_in(up))

    # 取 top_k
    top = day.sort("signal", descending=True).head(args.top_k)

    # join 最新收盘价 + 名称
    names = load_names()
    rows: list[dict] = []
    for i, r in enumerate(top.iter_rows(named=True), 1):
        vt = r["vt_symbol"]
        code = code_of(vt)
        # 最新收盘价
        pf = LAB_PATH / "daily" / f"{vt}.parquet"
        close = None
        if pf.exists():
            bar = pl.read_parquet(pf).filter(pl.col("datetime") == target)
            if not bar.is_empty():
                close = bar["close"].item(0)
        rows.append({
            "rank": i, "code": code, "name": names.get(code, "?"),
            "vt_symbol": vt, "signal": round(r["signal"], 5),
            "close": round(close, 2) if close else None,
        })

    out_df = pl.DataFrame(rows)
    print(f"\n=== {target.date()} 收盘后选股清单(次日开盘买入,top{args.top_k})===")
    print(f"模型: {args.name} | 剔除当日涨停 {len(up)} 只")
    print(out_df.to_pandas().to_string(index=False))

    csv_path = LAB_PATH / f"picks_{target.strftime('%Y%m%d')}.csv"
    out_df.write_csv(str(csv_path))
    print(f"\n清单已存: {csv_path}")
    print("⚠️ 研究输出,非投资建议;实盘需定期用最新数据重训模型")


if __name__ == "__main__":
    main()
