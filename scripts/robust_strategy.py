"""
健壮版选股策略 — 修复 EquityDemoStrategy 买入端的停牌日 KeyError

EquityDemoStrategy.on_bars 买入端(line 92 bars[vt_symbol])未检查股票当天是否有
bar。T+1 信号平移后，signal 日期可能落在某股停牌日(无 bar)，触发 KeyError。
本类重写 on_bars，在选股前过滤掉当天无 bar 的股票(卖出端父类已有 .get 保护)。
逻辑与父类完全一致，仅增加一行可交易过滤。
"""

import polars as pl

from vnpy.trader.object import BarData
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy


class RobustEquityStrategy(EquityDemoStrategy):
    """过滤当天无 bar 股票的健壮版长多策略,支持 N 日调仓周期"""

    rebalance_days: int = 1  # 每 N 个交易日调仓一次(1=每日);非调仓日持仓不动,让 N 日 alpha 兑现

    def on_init(self) -> None:
        super().on_init()
        self._rebal_count: int = 0

    def on_bars(self, bars: dict[str, BarData]) -> None:
        # 调仓周期控制:非调仓日保持持仓不动
        self._rebal_count += 1
        if self.rebalance_days > 1 and (self._rebal_count - 1) % self.rebalance_days != 0:
            return

        # 获取最新信号并排序
        last_signal: pl.DataFrame = self.get_signal()
        last_signal = last_signal.sort("signal", descending=True)

        # 关键修复：只保留当天有 bar(可交易)的股票，避免停牌日买入 KeyError
        tradable: list[str] = list(bars.keys())
        last_signal = last_signal.filter(pl.col("vt_symbol").is_in(tradable))

        # 更新持仓天数
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]
        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        # 卖出列表
        active_symbols: set[str] = set(last_signal["vt_symbol"][:self.top_k])
        active_symbols.update(pos_symbols)
        active_df: pl.DataFrame = last_signal.filter(pl.col("vt_symbol").is_in(active_symbols))

        component_symbols: set[str] = set(last_signal["vt_symbol"])
        sell_symbols: set[str] = set(pos_symbols).difference(component_symbols)

        for vt_symbol in active_df["vt_symbol"][-self.n_drop:]:
            if vt_symbol in pos_symbols:
                sell_symbols.add(vt_symbol)

        # 买入列表
        buyable_df: pl.DataFrame = last_signal.filter(~pl.col("vt_symbol").is_in(pos_symbols))
        buy_quantity: int = len(sell_symbols) + self.top_k - len(pos_symbols)
        buy_symbols: list = list(buyable_df[:buy_quantity]["vt_symbol"])

        # 卖出调仓
        cash: float = self.get_cash_available()
        for vt_symbol in sell_symbols:
            if self.holding_days[vt_symbol] < self.min_days:
                continue
            bar: BarData | None = bars.get(vt_symbol)
            if not bar:
                continue
            sell_price: float = bar.close_price
            sell_volume: float = self.get_pos(vt_symbol)
            self.set_target(vt_symbol, target=0)
            turnover: float = sell_price * sell_volume
            cost: float = max(turnover * self.close_rate, self.min_commission)
            cash += turnover - cost

        # 买入调仓(buy_symbols 已保证当天有 bar)
        if buy_symbols:
            buy_value: float = cash * self.cash_ratio / len(buy_symbols)
            for vt_symbol in buy_symbols:
                buy_price: float = bars[vt_symbol].close_price
                if not buy_price:
                    continue
                from vnpy.trader.utility import round_to
                buy_volume: float = round_to(buy_value / buy_price, self.min_volume)
                self.set_target(vt_symbol, buy_volume)

        # 执行交易
        self.execute_trading(bars, price_add=self.price_add)
