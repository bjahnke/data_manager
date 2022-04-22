from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import typing as t
import pandas_accessor.pda as pda


class StockDataGetter:
    def __init__(self, data_getter_method: t.Callable):
        self._data_getter_method = data_getter_method
        self.strategy_exceptions = []
        self.no_data = []

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        data = self._data_getter_method(symbol)
        if data is None:
            return pd.DataFrame()
        # data = data.rename(
        #     columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        # )
        pda.PriceTable(data, symbol)
        return data[["open", "high", "low", "close"]]

    def yield_strategy_data(
        self,
        bench_symbol,
        symbols: t.List[str],
        strategy: t.Callable[[pd.DataFrame, pd.DataFrame], t.Any],
        expected_exceptions: t.Tuple
    ) -> t.Union[t.Tuple[pd.DataFrame, t.Any], t.Tuple[None, None]]:
        """

        :param bench_symbol:
        :param symbols:
        :param strategy:
        :param expected_exceptions: t.Union[t.Tuple[pd.DataFrame, sfcr.FcStrategyTables], t.Tuple[None, None]]
        :return:
        """
        for i, symbol in enumerate(symbols):
            print(f"({i}/{len(symbols)}) {symbol}")  # , end='\r')
            bench_data = self.get_stock_data(bench_symbol)
            symbol_data = self.get_stock_data(symbol)
            strategy_data = None
            if not symbol_data.empty:
                try:
                    strategy_data = strategy(symbol_data, bench_data)
                except expected_exceptions as e:
                    self.strategy_exceptions.append(e)
            else:
                self.no_data.append(symbol)
                symbol_data = None

            yield symbol, symbol_data, bench_data, strategy_data


def enhanced_price_data_plot(data, ax=None):
    _open = "open"
    _high = "high"
    _low = "low"
    _close = "close"
    a = data[
        [_close, "hi3", "lo3", "clg", "flr", "rg_ch", "hi2", "lo2", 'stop_loss', 'french_stop']
    ].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:", "r.", "g."],
        figsize=(15, 5),
        grid=True,
        use_index=False,
        ax=ax
    )

    data["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=False,
    )


def plot(_d, _plot_window=0, _use_index=False, _axis=None, _ticker=None):
    """"""
    cols = [
        'close',
        "hi3",
        "lo3",
        "clg",
        "flr",
        # "rg_ch",
        "rg",
        'hi2_lag',
        'hi3_lag',
        'lo2_lag',
        'lo3_lag',
    ]
    style = ["grey", "ro", "go", "kv", "k^"]
    # if _axis is None:
    #     figure, _axis = plt.subplots(2, 1)
    _axis = (
        _d[cols]
            .iloc[_plot_window:]
            .plot(
                style=style,
                figsize=(15, 5),
                secondary_y=["rg"],
                # grid=True,
                # title=str.upper(_ticker),
                use_index=_use_index,
                ax=_axis
            )
    )

    return _axis


def update_sw_lag(swing_lags: pd.Series, swings: pd.Series, discovered_sw_dates):
    """
    Note this function updates args passed for swing_lags and discovered_sw_dates
    :param swing_lags:
    :param swings:
    :param discovered_sw_dates:
    :return:
    """
    swing_lags = swing_lags.reindex(swings.index)
    latest_sw = swings.loc[~pd.isna(swings)].iloc[-1:]
    if latest_sw.index not in discovered_sw_dates:
        swing_lags.loc[latest_sw.index[0]:] = latest_sw[0]
        discovered_sw_dates.append(latest_sw.index)

    return swing_lags


def rolling_plot(
    price_data: pd.DataFrame,
    ndf,
    stop_loss_t,
    peak_table,
    ticker,
    strategy_simulator: t.Callable,
    plot_rolling_lag=True,
    plot_loop=True,
    expected_exceptions=None
):
    """
    expected exceptions: src.utils.regime.NotEnoughDataError, src.floor_ceiling_regime.NoEntriesError
    recalculates the strategy on a rolling window of the given data to visualize how
    the strategy behaves
    """
    if expected_exceptions is None:
        expected_exceptions = ()
    _open = "open"
    _high = "high"
    _low = "low"
    _close = "close"
    use_index = False
    initial_size = 200  # 386
    plot_window = 250
    axis = None
    index = initial_size
    fp_rg = None
    hi2_lag = None
    hi3_lag = None
    lo2_lag = None
    lo3_lag = None
    hi2_discovery_dts = []
    hi3_discovery_dts = []
    lo2_discovery_dts = []
    lo3_discovery_dts = []

    pt = pda.PeakTable(peak_table)
    pt.data['px'] = pt.start_price(ndf)
    pt = pt.unpivot(ndf.index)
    _shi_px = pt.loc[(pt.type == -1)]
    _shi_2 = _shi_px.loc[(_shi_px.lvl == 2)]
    _shi_3 = _shi_px.loc[(_shi_px.lvl == 3)]

    _slo_px = pt.loc[(pt.type == 1)]
    _slo_2 = _slo_px.loc[(_slo_px.lvl == 2)]
    _slo_3 = _slo_px.loc[(_slo_px.lvl == 3)]

    ndf['hi2_lag'] = _shi_2.px.copy()
    ndf['hi3_lag'] = _shi_3.px.copy()
    ndf['lo2_lag'] = _slo_2.px.copy()
    ndf['lo3_lag'] = _slo_3.px.copy()

    d = ndf[[_open, _high, _low, _close, 'hi2_lag', 'hi3_lag', 'lo2_lag', 'lo3_lag']].copy().iloc[:index]

    ndf["stop_loss"] = stop_loss_t
    # a = ndf[
    #     [_close, "hi3", "lo3", "clg", "flr", "rg_ch", "hi2", "lo2", "stop_loss"]
    # ].plot(
    #     style=["grey", "ro", "go", "kv", "k^", "c:", "r.", "g."],
    #     figsize=(15, 5),
    #     # grid=True,
    #     title=str.upper(ticker),
    #     use_index=use_index,
    # )
    #
    # ndf["rg"].plot(
    #     style=["b-."],
    #     # figsize=(15, 5),
    #     # marker='o',
    #     secondary_y=["rg"],
    #     ax=a,
    #     use_index=use_index,
    # )
    # plt.show()
    # all_retest_swing(data, 'rt', distance_percent, retrace_percent, swing_window)
    # data[['close', 'hi3', 'lo3', 'rt']].plot(
    #     style=['grey', 'rv', 'g^', 'ko'],
    #     figsize=(10, 5), grid=True, title=str.upper(ticker))

    # data[['close', 'hi3', 'lo3']].plot(
    #     style=['grey', 'rv', 'g^'],
    #     figsize=(20, 5), grid=True, title=str.upper(ticker))

    # plt.show()
    """
    ohlc = ['Open','High','Low','Close']
    _o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]
    rg_val = ['Hi3','Lo3','flr','clg','rg','rg_ch',1.5]
    slo, shi,flr,clg,rg,rg_ch,threshold = [rg_val[s] for s in range(len(rg_val))]
    stdev = df[_c].rolling(63).std(ddof=0)
    df = regime_floor_ceiling(df,_h,_l,_c,slo, shi,flr,clg,rg,rg_ch,stdev,threshold)

    df[[_c,'Hi3', 'Lo3','clg','flr','rg_ch','rg']].plot(    
    style=['grey', 'ro', 'go', 'kv', 'k^','c:','y-.'],     
    secondary_y= ['rg'],figsize=(20,5),    
    grid=True, 
    title = str.upper(ticker))

    """

    new_axis = True

    for idx, row in ndf.iterrows():
        if (num := ndf.index.get_loc(idx)) <= index:
            print(f"iter index {num}")
            continue
        d.at[idx] = row
        try:
            res = strategy_simulator(d)
            # rt_copy = d.rt.copy()
            d = res.enhanced_price_data
            # d.rt.loc[pd.notna(rt_copy)] = rt_copy.loc[pd.notna(rt_copy)]

            if fp_rg is None:
                fp_rg = d.rg.copy()
                fp_rg = fp_rg.fillna(0)
                hi2_lag = d.hi2.copy()
                hi3_lag = d.hi3.copy()
                lo2_lag = d.lo2.copy()
                lo3_lag = d.lo3.copy()
            else:
                fp_rg = fp_rg.reindex(d.rg.index)
                fp_rg.iloc[-1] = d.rg.iloc[-1]
                hi2_lag = update_sw_lag(hi2_lag, d.hi2, hi2_discovery_dts)
                hi3_lag = update_sw_lag(hi3_lag, d.hi3, hi3_discovery_dts)
                lo2_lag = update_sw_lag(lo2_lag, d.lo2, lo2_discovery_dts)
                lo3_lag = update_sw_lag(lo3_lag, d.lo3, lo3_discovery_dts)
        except (KeyError, IndexError) + expected_exceptions:
            print(f"iter index {num}")
            pass
        else:
            # live print procedure
            try:
                if plot_loop is True:
                    data_plot_window = len(d.index) - plot_window
                    data_plot_window = data_plot_window if data_plot_window >= 0 else 0
                    if new_axis is False:
                        plt.gca().cla()
                        axis.clear()
                        main_plot_window = data_plot_window
                    else:
                        main_plot_window = index - plot_window

                    current_axis = plot(
                        _ticker=ticker,
                        _d=d,
                        _plot_window=main_plot_window,
                        _use_index=use_index,
                        _axis=axis
                    )

                    fp_rg.iloc[data_plot_window:].plot(
                        style="y-.", secondary_y=True, use_index=use_index, ax=axis
                    )

                    if plot_rolling_lag is True:
                        hi2_lag.iloc[data_plot_window:].plot(
                            style="ro", use_index=use_index, ax=axis
                        )
                        lo2_lag.iloc[data_plot_window:].plot(
                            style="go", use_index=use_index, ax=axis
                        )
                        hi3_lag.iloc[data_plot_window:].plot(
                            style="r.", use_index=use_index, ax=axis
                        )
                        lo3_lag.iloc[data_plot_window:].plot(
                            style="g.", use_index=use_index, ax=axis
                        )

                    if new_axis is True:
                        new_axis = False
                        axis = current_axis
                        plt.ion()
                        plt.show()

                    plt.pause(0.001)
            except Exception as e:
                print(e)
        print(idx)

    if plot_loop is True:
        plt.close()

    a = d[[_close, "hi3", "lo3", "clg", "flr", "rg_ch"]].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:"],
        figsize=(15, 5),
        # grid=True,
        title=str.upper(ticker),
        use_index=use_index,
    )
    d["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=use_index,
    )
    fp_rg.plot(style="y-.", secondary_y=True, use_index=use_index, ax=a)
    hi2_lag.plot(style="ro", use_index=use_index, ax=a)
    hi3_lag.plot(style="r.", use_index=use_index, ax=a)
    lo2_lag.plot(style="go", use_index=use_index, ax=a)
    lo3_lag.plot(style="g.", use_index=use_index, ax=a)
    plt.show()

    return


def yf_get_stock_data(symbol, days, interval: str) -> pd.DataFrame:
    """get price data from yahoo finance"""
    data = yf.ticker.Ticker(symbol).history(
        start=(datetime.now() - timedelta(days=days)),
        end=datetime.now(),
        interval=interval,
    )
    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    pda.PriceTable(data, symbol)
    data = data[["open", "high", "low", "close"]]
    return data


def get_cached_data(symbol, days, interval) -> pd.DataFrame:
    """get price data from local storage"""
    file_name = f"{symbol}_{interval}_{days}d.csv"
    data = pd.read_csv(rf"..\strategy_output\price_data\{file_name}")
    data_cols = data.columns.to_list()

    if "Date" in data_cols:
        date_col = "Date"
    elif "Datetime" in data_cols:
        date_col = "Datetime"
    else:
        raise

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(data[date_col])

    return data


def data_to_relative(data, bench_df):
    """make relative function compatible with my code"""
    bm_close = 'bm_close'
    bench_df = bench_df.rename(columns={'close': bm_close})
    working_data = regime.relative(
        df=data, bm_df=bench_df, bm_col=bm_close, rebase=True
    )
    working_data = working_data[["ropen", "rhigh", "rlow", "rclose"]]
    working_data = working_data.rename(
        columns={
            "ropen": "open",
            "rhigh": "high",
            "rlow": "low",
            "rclose": "close",
        }
    )
    return working_data


def get_wikipedia_stocks(url):
    """get stock data (names, sectors, etc) from wikipedia"""
    wiki_df = pd.read_html(url)[0]
    tickers_list = list(wiki_df["Symbol"])
    return tickers_list[:], wiki_df


def yf_download_data(tickers, days, interval) -> pd.DataFrame:
    """download stock data from yf and concat to big price history file"""
    data = yf.download(
        tickers,
        start=(datetime.now() - timedelta(days=days)),
        end=datetime.now(),
        interval=interval
    )

    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    return data[["open", "high", "low", "close"]]


@dataclass
class ScanData:
    stat_overview: pd.DataFrame
    strategy_lookup: t.Dict
    entry_table: pd.DataFrame
    peak_table: pd.DataFrame


def run_scanner(scanner, stat_calculator, restrict_side=False) -> ScanData:
    stat_overview = pd.DataFrame()
    entry_data = {}
    strategy_data_lookup = {}
    entry_table = []
    peak_table = []
    for symbol, symbol_data, bench_data, strategy_data in scanner:
        if symbol_data is None or strategy_data is None:
            continue

        signals = strategy_data.valid_entries.copy()

        if restrict_side is not False:
            signals = signals.loc[signals.dir == restrict_side].reset_index(drop=True)

        if signals.empty:
            continue

        # stat_sheet_historical = stat_calculator(symbol_data, signals)
        stat_sheet_historical = stat_calculator(strategy_data.enhanced_price_data, signals)

        if stat_sheet_historical is None:
            continue

        strategy_data.stat_historical = stat_sheet_historical

        strategy_data_lookup[symbol] = strategy_data
        signals['symbol'] = symbol

        strategy_data.peak_table['symbol'] = symbol
        peak_table.append(strategy_data.peak_table)

        stat_sheet_final_scores = stat_sheet_historical.iloc[-1].copy()
        stat_sheet_final_scores['symbol'] = symbol
        signal_table = pda.SignalTable(signals)
        price_table = pda.PriceTable(symbol_data, '')

        signals['abs_entry'] = signal_table.entry_prices(price_table)
        signals['abs_exit'] = signal_table.exit_prices(price_table)
        # signals['abs_return'] = signal_table.static_returns(price_table)
        signals['partial_exit'] = signal_table.partial_exit_prices(price_table)

        risk = signal_table.pyramid_all(-0.0075)

        # signals['win_roll'] = stat_sheet_historical.win_roll.loc[signals.entry].values
        # signals['avg_win_roll'] = stat_sheet_historical.avg_win_roll.loc[signals.entry].values
        # signals['avg_loss_roll'] = stat_sheet_historical.avg_loss_roll.loc[signals.entry].values
        signals['risk'] = risk

        # win_rate_query = (signals.win_roll > .5)
        # signals.loc[win_rate_query, 'risk'] = smm.other_kelly(
        #     win_rate=signals['win_roll'].loc[win_rate_query],
        #     avg_win=signals['avg_win_roll'].loc[win_rate_query],
        #     avg_loss=signals['avg_win_roll'].loc[win_rate_query],
        # )
        # signals['risk'] = signals['risk'] * .5

        r_multiplier = 1.5

        signals['shares'] = signal_table.eqty_risk_shares(strategy_data.enhanced_price_data, 30000, signals['risk'])

        partial_exit_ptc = ((signals.shares / r_multiplier) / signals.shares)
        remaining_exit_ptc = 1 - partial_exit_ptc
        partial_exit_shares = (signals.shares * partial_exit_ptc).apply(np.ceil)
        remaining_exit_shares = signals.shares - partial_exit_shares

        signals['partial_profit'] = (signals.partial_exit - signals.abs_entry) * partial_exit_shares
        signals['rem_profit'] = (signals.abs_exit - signals.abs_entry) * remaining_exit_shares
        signals['partial_total'] = signals.partial_profit + signals.rem_profit
        signals['no_partial_total'] = (signals.abs_exit - signals.abs_entry) * signals.shares
        signals['my_total'] = signals['partial_total']
        signals.loc[pd.isna(signals.my_total), 'my_total'] = signals.loc[pd.isna(signals.my_total), 'no_partial_total']
        signals['total'] = signals.my_total.cumsum()

        entry_data[symbol] = signals

        # fig, axes = plt.subplots(nrows=3, ncols=1)

        # strategy_data.enhanced_price_data['stop_loss'] = strategy_data.stop_loss_series
        # strategy_data.enhanced_price_data['french_stop'] = strategy_data.french_stop.stop_price
        # enhanced_price_data_plot(strategy_data.enhanced_price_data, ax=axes[0])

        # bench_data["close"] = bench_data["close"].div(bench_data["close"][0])
        # symbol_data['stop_loss'] = strategy_data.enhanced_price_data['stop_loss'] * bench_data.close
        # symbol_data[['close', 'stop_loss']].plot(use_index=False, ax=axes[1])
        # pd.DataFrame({
        #     'abs_rel_delta': abs(symbol_data.close - strategy_data.enhanced_price_data.close),
        #     '': abs(symbol_data.close - symbol_data.stop_loss)
        # }).plot(use_index=False, ax=axes[2])
        # plt.show()
        stat_sheet_final_scores['weight_total'] = signals.total.iloc[-1]
        stat_overview = pd.concat([stat_overview, stat_sheet_final_scores.to_frame().transpose()], ignore_index=True)
        entry_table.append(signals)

    stat_overview = stat_overview.reset_index(drop=True)
    entry_table = pd.concat(entry_table)
    peak_table = pd.concat(peak_table)
    return ScanData(stat_overview, strategy_data_lookup, entry_table, peak_table)


if __name__ == '__main__':
    print('here')
