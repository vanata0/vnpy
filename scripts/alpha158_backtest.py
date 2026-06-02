"""
选股交易回测(Phase 5)

用法：
    source /home/oracle/vnpy-venv/bin/activate
    python scripts/alpha158_backtest.py --name a158_mktcap500
    python scripts/alpha158_backtest.py --name a158_mktcap500 --filter-limit   # 启用涨停过滤

输出：策略年化/Sharpe/最大回撤 + 对标沪深300 的超额收益。

涨跌停过滤(--filter-limit)：剔除当日涨停股的 signal(买不进)。
MVP 边界：跌停卖不出未特殊处理(已在 plan 声明)。
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

from robust_strategy import RobustEquityStrategy

LAB_PATH = Path(__file__).resolve().parents[1] / "lab_data"
BENCHMARK = "000300.SSE"

OOS_START = datetime(2025, 1, 1)
OOS_END = datetime(2026, 5, 29)

# A股交易成本: 买入佣金万5; 卖出佣金万5+印花税千1; 股票 size=1, pricetick=0.01
LONG_RATE = 0.0005
SHORT_RATE = 0.0015


def shift_signal_t1(signal: pl.DataFrame) -> pl.DataFrame:
    """
    将 signal 平移到下一交易日，消除未来函数。

    引擎用开盘价成交(trade_price=min(order_price, open))。原始 signal[t] 基于
    t 日收盘后的因子，若在 t 日开盘成交即用了未来信息。平移后 signal[t] 标记为
    t+1，在 t+1 开盘执行 —— 收盘出信号、次日开盘交易，真实可实现。
    """
    dates = signal["datetime"].unique().sort().to_list()
    map_df = pl.DataFrame({
        "datetime": dates[:-1],
        "_next": dates[1:],
    }).with_columns(pl.col("datetime").cast(signal["datetime"].dtype),
                    pl.col("_next").cast(signal["datetime"].dtype))
    return (
        signal.join(map_df, on="datetime", how="inner")
        .drop("datetime")
        .rename({"_next": "datetime"})
        .select(["datetime", "vt_symbol", "signal"])
    )


def apply_limit_filter(signal: pl.DataFrame) -> pl.DataFrame:
    """剔除当日涨停股的 signal 行(买不进)"""
    limit = pl.read_parquet(LAB_PATH / "limit_status.parquet")
    up = limit.filter(pl.col("is_limitup")).select(["datetime", "vt_symbol"])
    return signal.join(up, on=["datetime", "vt_symbol"], how="anti")


def apply_regime_filter(signal: pl.DataFrame, lab: AlphaLab, ma_window: int) -> pl.DataFrame:
    """
    市场择时门控：沪深300 跌破 MA 的次日空仓(清空当日 signal → 策略清仓)。
    无前瞻：t 日收盘判断 risk_on，平移一日对齐执行日 t+1，与 signal T+1 口径一致。
    """
    smin = signal["datetime"].min()
    smax = signal["datetime"].max()
    bars = lab.load_bar_data(BENCHMARK, Interval.DAILY,
                             smin - timedelta(days=ma_window * 2 + 30), smax)
    bench = pl.DataFrame({
        "datetime": [b.datetime for b in bars],
        "close": [b.close_price for b in bars],
    }).sort("datetime")
    bench = bench.with_columns(pl.col("close").rolling_mean(ma_window).alias("ma"))
    bench = bench.with_columns((pl.col("close") > pl.col("ma")).alias("risk_on"))
    bench = bench.with_columns(pl.col("risk_on").shift(1).alias("risk_on_exec"))
    risk_off = bench.filter(pl.col("risk_on_exec") == False)["datetime"]  # noqa: E712

    total_days = signal["datetime"].n_unique()
    filtered = signal.filter(~pl.col("datetime").is_in(risk_off))
    kept = filtered["datetime"].n_unique()
    print(f"市场择时(沪深300 MA{ma_window}): 空仓 {total_days - kept}/{total_days} 交易日，持仓 {kept} 日")
    return filtered


def write_contracts(lab: AlphaLab, vt_symbols: list[str]) -> None:
    """一次性写入所有股票的合约配置(避免逐只 add_contract_setting 的 O(n^2) IO)"""
    contracts = {
        s: {"long_rate": LONG_RATE, "short_rate": SHORT_RATE, "size": 1, "pricetick": 0.01}
        for s in vt_symbols
    }
    with open(lab.contract_path, "w", encoding="UTF-8") as f:
        json.dump(contracts, f, indent=2)


def benchmark_excess(lab: AlphaLab, daily_df: pl.DataFrame, capital: int, start: datetime, end: datetime) -> None:
    """对标沪深300 计算超额收益(daily_df 仅含 *_pnl 列，用 net_pnl 累计)"""
    bars = lab.load_bar_data(BENCHMARK, Interval.DAILY, start, end)
    if not bars:
        print("benchmark 数据缺失，跳过超额计算")
        return

    strat_total = daily_df["net_pnl"].sum() / capital
    bench_total = bars[-1].close_price / bars[0].close_price - 1
    print(f"\n=== 对标沪深300 超额({start.date()}~{end.date()})===")
    print(f"策略总收益(净):  {strat_total:+.2%}")
    print(f"沪深300同期:     {bench_total:+.2%}")
    print(f"超额收益:        {strat_total - bench_total:+.2%}")


def save_curve(name: str, daily_df: pl.DataFrame, lab: AlphaLab, capital: int,
               start: datetime, end: datetime) -> None:
    """存净值曲线(策略 vs 沪深300，归一化到 1.0)供 dashboard 展示"""
    results_dir = LAB_PATH / "results"
    results_dir.mkdir(exist_ok=True)
    df = daily_df.sort("date")
    nav = ((capital + df["net_pnl"].cum_sum()) / capital).to_list()
    dates = df["date"].to_list()

    bars = lab.load_bar_data(BENCHMARK, Interval.DAILY, start, end)
    bmap = {b.datetime.date(): b.close_price for b in bars}
    bench_raw = [bmap.get(d) for d in dates]
    b0 = next((x for x in bench_raw if x), None)
    bench = [x / b0 if (x and b0) else None for x in bench_raw]

    out = results_dir / f"{name}_curve.parquet"
    pl.DataFrame({"date": dates, "strat": nav, "bench": bench}).write_parquet(str(out))
    print(f"净值曲线已存: {out.name}")


def run(name: str, top_k: int, n_drop: int, min_days: int, capital: int, filter_limit: bool, t1: bool,
        regime: bool = False, ma_window: int = 60,
        start: datetime = OOS_START, end: datetime = OOS_END) -> None:
    lab = AlphaLab(str(LAB_PATH))
    signal = lab.load_signal(name)
    if signal is None:
        print(f"ERROR: 找不到信号 {name}")
        sys.exit(1)

    if t1:
        signal = shift_signal_t1(signal)
        print("T+1 平移: 信号延迟到次日开盘执行(消除未来函数)")

    if regime:
        signal = apply_regime_filter(signal, lab, ma_window)

    if filter_limit:
        before = len(signal)
        signal = apply_limit_filter(signal)
        print(f"涨停过滤: {before:,} → {len(signal):,} 行")

    vt_symbols = signal["vt_symbol"].unique().to_list()
    print(f"回测股票池: {len(vt_symbols)} 只 | top_k={top_k} n_drop={n_drop} min_days={min_days} "
          f"capital={capital:,}")

    write_contracts(lab, vt_symbols)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=capital,
    )
    engine.add_strategy(
        RobustEquityStrategy,
        {"top_k": top_k, "n_drop": n_drop, "min_days": min_days},
        signal,
    )
    engine.load_data()
    engine.run_backtesting()
    daily_df = engine.calculate_result()
    if daily_df is None:
        print("回测无成交记录")
        return
    engine.calculate_statistics()

    benchmark_excess(lab, daily_df, capital, start, end)
    save_curve(name, daily_df, lab, capital, start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description="选股交易回测")
    parser.add_argument("--name", default="a158_mktcap500", help="信号名称")
    parser.add_argument("--top-k", type=int, default=50, help="最大持仓数")
    parser.add_argument("--n-drop", type=int, default=5, help="每次换仓卖出数")
    parser.add_argument("--min-days", type=int, default=3, help="最短持有天数")
    parser.add_argument("--capital", type=int, default=100_000_000, help="初始资金")
    parser.add_argument("--filter-limit", action="store_true", help="启用涨停过滤")
    parser.add_argument("--t1", action="store_true", help="T+1 开盘成交(消除未来函数)")
    parser.add_argument("--regime", action="store_true", help="市场择时门控(沪深300 跌破 MA 空仓)")
    parser.add_argument("--ma-window", type=int, default=60, help="择时均线窗口")
    parser.add_argument("--start", default=None, help="回测起始日(YYYY-MM-DD)，默认 2025-01-01")
    parser.add_argument("--end", default=None, help="回测结束日(YYYY-MM-DD)，默认 2026-05-29")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else OOS_START
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else OOS_END

    run(
        name=args.name,
        top_k=args.top_k,
        n_drop=args.n_drop,
        min_days=args.min_days,
        capital=args.capital,
        filter_limit=args.filter_limit,
        t1=args.t1,
        regime=args.regime,
        ma_window=args.ma_window,
        start=start,
        end=end,
    )


if __name__ == "__main__":
    main()
