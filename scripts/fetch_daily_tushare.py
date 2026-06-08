"""
从 Tushare 拉取 A 股日线前复权数据 → AlphaLab daily parquet

替代 alpha158_bridge.py，不再依赖 stock_new DuckDB。

两种模式：
    增量（默认）— 每新交易日仅 ~2 次 API 调用，秒级完成
        python scripts/fetch_daily_tushare.py

    全量重载 — 首次建库或全量复权重算
        python scripts/fetch_daily_tushare.py --full

增量原理：
    先用 pro.daily(trade_date=D) 按日批量拉所有股票（1 次 API）；
    再用 pro.adj_factor(trade_date=D) 获取当日复权因子（1 次 API）；
    取 close_qfq = close_raw（最新日期前复权 = 当日收盘价本身），
    对历史行用 close_raw * adj_today / adj_hist 重新对齐。

    简化处理：当日无新除权时（绝大多数情况）adj_today=adj_hist=不变，
    直接 append 即可。有除权的股票检测到 adj 跳变后触发单股全量重取。
"""

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tushare_client import pro, ts
from vnpy.alpha import AlphaLab

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
OUT_DIR = LAB_PATH / "daily"
UNIVERSE = "CSI500"
FULL_START = "20180101"


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def ts_code_to_vt(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    if exchange == "SH":
        return f"{code}.SSE"
    if exchange == "SZ":
        return f"{code}.SZSE"
    if exchange == "BJ":
        return f"{code}.BSE"
    raise ValueError(f"未知交易所: {ts_code}")


def vt_to_ts_code(vt_symbol: str) -> str:
    code, exchange = vt_symbol.rsplit(".", 1)
    if exchange == "SSE":
        return f"{code}.SH"
    if exchange == "SZSE":
        return f"{code}.SZ"
    if exchange == "BSE":
        return f"{code}.BJ"
    raise ValueError(f"未知 exchange: {vt_symbol}")


def limit_pct(ts_code: str) -> float:
    """根据股票代码推算涨跌停幅度（百分比）"""
    code = ts_code.split(".")[0]
    if code.startswith("688"):
        return 19.9   # 科创板
    if code.startswith("30"):
        return 19.9   # 创业板（2020 年后注册制）
    if code.startswith(("4", "8")):
        return 29.9   # 北交所
    return 9.9        # 主板（含 ST 近似，ST 实际 4.9% 但较少影响选股）


def bar_to_parquet_row(row: dict) -> dict:
    """把 Tushare pro_bar 的一行转成 parquet 字段"""
    vol = float(row.get("vol") or 0)
    amount = float(row.get("amount") or 0)
    volume = vol * 100           # 手 → 股
    turnover = amount * 1000     # 千元 → 元
    vwap = (turnover / volume) if volume > 0 else float(row.get("close") or 0)
    return {
        "open":         float(row.get("open") or 0),
        "high":         float(row.get("high") or 0),
        "low":          float(row.get("low") or 0),
        "close":        float(row.get("close") or 0),
        "volume":       volume,
        "turnover":     turnover,
        "vwap":         vwap,
        "open_interest": 0.0,
    }


# ─── 全量重载（per-stock pro_bar）─────────────────────────────────────────────

def full_reload(symbols: list[str]) -> None:
    """全量拉取所有股票的历史前复权数据"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(symbols)
    print(f"全量重载：{total} 只股票，起始日 {FULL_START}", flush=True)

    ok = fail = 0
    for i, vt in enumerate(symbols, 1):
        ts_code = vt_to_ts_code(vt)
        try:
            df = ts.pro_bar(api=pro, ts_code=ts_code, adj="qfq",
                            start_date=FULL_START, freq="D")
            if df is None or df.empty:
                fail += 1
                continue
            _write_parquet(vt, df)
            ok += 1
        except Exception as e:
            print(f"  ⚠️  {vt}: {e}", flush=True)
            fail += 1
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] ok={ok} fail={fail}", flush=True)
        time.sleep(0.12)   # ~8 req/s，避免限频

    print(f"\n全量完成: ok={ok} fail={fail}", flush=True)


def _write_parquet(vt: str, df) -> None:
    """把 Tushare DataFrame 转成 parquet 写盘"""
    rows = []
    for row in df.to_dict("records"):
        try:
            dt = pl.Series([row["trade_date"]]).str.to_datetime("%Y%m%d")[0]
        except Exception:
            continue
        r = bar_to_parquet_row(row)
        r["datetime"] = dt
        rows.append(r)
    if not rows:
        return
    out_df = pl.DataFrame(rows).select([
        "datetime", "open", "high", "low", "close",
        "volume", "turnover", "vwap", "open_interest",
    ]).sort("datetime")
    path = OUT_DIR / f"{vt}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    out_df.write_parquet(str(tmp))
    tmp.replace(path)


# ─── 增量更新（per-date batch）────────────────────────────────────────────────

def incremental_update(symbols: list[str]) -> None:
    """
    检测每只股票 parquet 的最新日期，找出需要更新的新交易日，
    用 pro.daily(trade_date=D) 批量拉新日期，再追加各 parquet。
    有除权的股票单独触发全量重取。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ① 扫描各 parquet 的最新日期
    last_dates: dict[str, str] = {}   # vt_symbol → "YYYYMMDD"
    for vt in symbols:
        p = OUT_DIR / f"{vt}.parquet"
        if p.exists():
            try:
                d = pl.read_parquet(p, columns=["datetime"])["datetime"].max()
                if d is not None:
                    last_dates[vt] = d.strftime("%Y%m%d")
            except Exception:
                pass

    if not last_dates:
        print("无已有 parquet，切换为全量重载", flush=True)
        full_reload(symbols)
        return

    # ② 找出所有股票中最早的 last_date（缺失 parquet 的股票触发单独全量）
    missing = [v for v in symbols if v not in last_dates]
    if missing:
        print(f"  {len(missing)} 只无 parquet，触发单独全量重取...", flush=True)
        full_reload(missing)
        for vt in missing:
            p = OUT_DIR / f"{vt}.parquet"
            if p.exists():
                d = pl.read_parquet(p, columns=["datetime"])["datetime"].max()
                if d is not None:
                    last_dates[vt] = d.strftime("%Y%m%d")

    # 以多数股票的最新日期作为增量起点
    from collections import Counter
    most_common_last = Counter(last_dates.values()).most_common(1)[0][0]
    today_str = time.strftime("%Y%m%d")

    # ③ 拿最新交易日历（从 pro.trade_cal）
    cal = pro.trade_cal(exchange="SSE", start_date=most_common_last, end_date=today_str,
                        is_open="1", fields="cal_date")
    if cal is None or cal.empty:
        print("无新交易日，数据已是最新", flush=True)
        return
    new_dates = sorted(cal["cal_date"].tolist())
    # 去掉已包含的 last_date 本身
    new_dates = [d for d in new_dates if d > most_common_last]
    if not new_dates:
        print("无新交易日，数据已是最新", flush=True)
        return
    print(f"  增量日期: {new_dates[0]} ~ {new_dates[-1]}（{len(new_dates)} 个交易日）", flush=True)

    # ④ 按日拉数据并更新 parquet
    limit_status_rows: list[dict] = []
    adj_cache: dict[str, float] = {}   # ts_code → 当日 adj_factor

    for trade_date in new_dates:
        print(f"  拉取 {trade_date}...", flush=True)

        # 获取当日全市场原始日线
        raw = pro.daily(trade_date=trade_date,
                        fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg")
        time.sleep(0.15)

        # 获取当日复权因子
        adj = pro.adj_factor(trade_date=trade_date, fields="ts_code,adj_factor")
        time.sleep(0.15)

        if raw is None or raw.empty:
            print(f"  ⚠️  {trade_date}: daily 为空，跳过", flush=True)
            continue

        adj_map: dict[str, float] = {}
        if adj is not None and not adj.empty:
            adj_map = dict(zip(adj["ts_code"], adj["adj_factor"].astype(float)))

        # 按 ts_code 索引当日数据
        day_map = {row["ts_code"]: row for row in raw.to_dict("records")}

        for vt in symbols:
            ts_code = vt_to_ts_code(vt)
            row = day_map.get(ts_code)
            if row is None:
                continue

            try:
                dt = pl.Series([trade_date]).str.to_datetime("%Y%m%d")[0]
            except Exception:
                continue

            # 前复权：raw close × adj_factor_today / adj_factor_today = raw close
            # （最新日期前复权价 = 当日收盘价本身；历史价已在 parquet 中以相对比例存储）
            adj_today = adj_map.get(ts_code, 1.0)
            adj_prev = adj_cache.get(ts_code)

            # 检测除权：adj_factor 发生变化 → 单股全量重取
            if adj_prev is not None and abs(adj_today - adj_prev) > 1e-6:
                print(f"  除权检测: {vt} adj {adj_prev:.4f}→{adj_today:.4f}，触发单股全量重取", flush=True)
                try:
                    df = ts.pro_bar(api=pro, ts_code=ts_code, adj="qfq",
                                    start_date=FULL_START, freq="D")
                    time.sleep(0.15)
                    if df is not None and not df.empty:
                        _write_parquet(vt, df)
                except Exception as e:
                    print(f"  ⚠️  {vt} 重取失败: {e}", flush=True)
                adj_cache[ts_code] = adj_today
                continue   # 已全量写入，跳过 append

            adj_cache[ts_code] = adj_today

            # 正常追加
            r = bar_to_parquet_row(row)
            r["datetime"] = dt
            new_row = pl.DataFrame([r]).select([
                "datetime", "open", "high", "low", "close",
                "volume", "turnover", "vwap", "open_interest",
            ])

            p = OUT_DIR / f"{vt}.parquet"
            if p.exists():
                old = pl.read_parquet(p)
                combined = pl.concat([old, new_row]).unique("datetime").sort("datetime")
            else:
                combined = new_row
            combined.write_parquet(str(p))

            # 收集 limit_status
            pct = float(row.get("pct_chg") or 0)
            lim = limit_pct(ts_code)
            limit_status_rows.append({
                "datetime": dt,
                "vt_symbol": vt,
                "is_limitup":   pct >= lim,
                "is_limitdown": pct <= -lim,
            })

        print(f"  ✓ {trade_date} 完成", flush=True)

    # ⑤ 更新 limit_status.parquet
    if limit_status_rows:
        _update_limit_status(limit_status_rows)

    total = len(symbols)
    print(f"\n增量完成：{total} 只，{len(new_dates)} 个新交易日", flush=True)


def _update_limit_status(new_rows: list[dict]) -> None:
    path = LAB_PATH / "limit_status.parquet"
    new_df = pl.DataFrame(new_rows)
    if path.exists():
        old = pl.read_parquet(path)
        combined = (
            pl.concat([old, new_df])
            .unique(["datetime", "vt_symbol"])
            .sort(["datetime", "vt_symbol"])
        )
    else:
        combined = new_df.sort(["datetime", "vt_symbol"])
    combined.write_parquet(str(path))
    print(f"limit_status.parquet 更新: {len(combined):,} 行", flush=True)


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="从 Tushare 拉取日线前复权数据")
    parser.add_argument("--full", action="store_true", help="全量重载（首次建库或全量复权重算）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 只（测试用）")
    args = parser.parse_args()

    lab = AlphaLab(str(LAB_PATH))
    # 取 CSI500 历史所有出现过的股票
    from scripts.alpha158_train import TRAIN_PERIOD, TEST_PERIOD
    symbols = lab.load_component_symbols(UNIVERSE, TRAIN_PERIOD[0], TEST_PERIOD[1])
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"universe={UNIVERSE}  股票池: {len(symbols)} 只", flush=True)
    t0 = time.monotonic()

    if args.full:
        full_reload(symbols)
    else:
        incremental_update(symbols)

    print(f"总耗时: {(time.monotonic()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
