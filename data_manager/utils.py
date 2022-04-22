# import src.utils.regime as regime
import data_manager.scanner as scanner
# import src.floor_ceiling_regime as sfcr
import json
import typing as t
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool, cpu_count
from time import perf_counter
import pandas_accessors.utils as pda_utils
import pandas_accessors.accessors as pda
import numpy as np


class DataLoader:
    def __init__(self, data: t.Dict, base_file_path: str):
        self._data = data
        self._base_file_path = base_file_path

    @classmethod
    def init_from_paths(cls, data_path, base_path):
        with open(data_path, 'r') as data_fp:
            data = json.load(data_fp)
        with open(base_path, 'r') as base_fp:
            base = json.load(base_fp)['base']
        return cls(data, base)

    @property
    def files(self):
        return self._data['files']

    @property
    def base(self):
        return self._base_file_path

    def file_path(self, file_name):
        return fr'{self.base}\{file_name}'

    def history_path(self, bench: str, interval: str):
        return self.file_path(self.files['history'][str(interval)][bench])

    def bench_path(self, bench: str, interval: str):
        return self.file_path(self.files['bench'][str(interval)][bench])


def simple_relative(df, bm_close, rebase=True):
    """simplified version of relative calculation"""
    bm = bm_close.ffill()
    if rebase is True:
        bm = bm.div(bm[0])
    return df.div(bm, axis=0)


def expand_index(gap_data, full_index):
    """insert indexes into the given gap data"""
    try:
        expanded_idx = gap_data.__class__(index=full_index, columns=gap_data.columns, dtype='float64')
    except AttributeError:
        expanded_idx = gap_data.__class__(index=full_index, dtype='float64')
    expanded_idx.loc[gap_data.index] = gap_data
    return expanded_idx


def calc_stats(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    min_periods: int,
    window: int,
    percentile: float,
    limit,
    round_to=2,
) -> t.Union[None, pd.DataFrame]:
    """
    get full stats of strategy, rolling and expanding
    :param round_to:
    :param signals:
    :param price_data:
    :param min_periods:
    :param window:
    :param percentile:
    :param limit:
    :return:
    """
    price_data = round(price_data, round_to)
    # TODO include regime returns
    signal_table = pda.SignalTable(signals.copy())
    signal_table.data["trade_count"] = signal_table.counts
    signals_un_pivot = signal_table.unpivot()
    signals_un_pivot = signals_un_pivot.loc[
        ~signals_un_pivot.index.duplicated(keep="last")
    ]
    signals_un_pivot = signals_un_pivot[['dir', 'trade_count']]
    signals_un_pivot = expand_index(signals_un_pivot, price_data.index)
    signals_un_pivot.dir = signals_un_pivot.dir.fillna(0)

    passive_returns_1d = pda_utils.simple_log_returns(price_data.close)
    signals_un_pivot["strategy_returns_1d"] = passive_returns_1d * signals_un_pivot.dir
    # don't use entry date to calculate returns
    signals_un_pivot.loc[signal_table.entry, "strategy_returns_1d"] = 0
    strategy_returns_1d = signals_un_pivot.strategy_returns_1d.copy()

    # Performance
    cumul_passive = pda_utils.cumulative_returns_pct(passive_returns_1d, min_periods)
    cumul_returns = pda_utils.cumulative_returns_pct(strategy_returns_1d, min_periods)
    cumul_excess = cumul_returns - cumul_passive - 1
    cumul_returns_pct = cumul_returns.copy()

    # Robustness metrics
    grit_expanding = pda_utils.expanding_grit(cumul_returns)
    grit_roll = pda_utils.rolling_grit(cumul_returns, window)

    tr_expanding = pda_utils.expanding_tail_ratio(cumul_returns, percentile, limit)
    tr_roll = pda_utils.rolling_tail_ratio(cumul_returns, window, percentile, limit)

    profits_expanding = pda_utils.expanding_profits(strategy_returns_1d)
    losses_expanding = pda_utils.expanding_losses(strategy_returns_1d)
    pr_expanding = pda_utils.profit_ratio(profits=profits_expanding, losses=losses_expanding)

    profits_roll = pda_utils.rolling_profits(strategy_returns_1d, window)
    losses_roll = pda_utils.rolling_losses(strategy_returns_1d, window)
    pr_roll = pda_utils.profit_ratio(profits=profits_roll, losses=losses_roll)

    # Cumulative t-stat
    win_count = (
        strategy_returns_1d.loc[strategy_returns_1d > 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    total_count = (
        strategy_returns_1d.loc[strategy_returns_1d != 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    csr_expanding = pda_utils.common_sense_ratio(pr_expanding, tr_expanding)
    csr_roll = pda_utils.common_sense_ratio(pr_roll, tr_roll)
    csr_roll = expand_index(csr_roll, price_data.index).ffill()

    # Trade Count
    trade_count = signals_un_pivot["trade_count"]
    trade_count = expand_index(trade_count, price_data.index).ffill().fillna(0)
    signal_roll = trade_count.diff(window)

    win_rate = (win_count / total_count).fillna(method="ffill")
    avg_win = profits_expanding / total_count
    avg_loss = losses_expanding / total_count
    edge_expanding = pda_utils.expectancy(win_rate, avg_win, avg_loss).fillna(method="ffill")
    sqn_expanding = pda_utils.t_stat(trade_count, edge_expanding)

    win_roll = strategy_returns_1d.copy()
    win_roll[win_roll <= 0] = np.nan
    win_rate_roll = win_roll.rolling(window, min_periods=0).count() / window
    avg_win_roll = profits_roll / window
    avg_loss_roll = losses_roll / window

    edge_roll = pda_utils.expectancy(
        win_rate=win_rate_roll, avg_win=avg_win_roll, avg_loss=avg_loss_roll
    )
    sqn_roll = pda_utils.t_stat(signal_count=signal_roll, trading_edge=edge_roll)

    score_expanding = pda_utils.robustness_score(grit_expanding, csr_expanding, sqn_expanding)
    score_roll = pda_utils.robustness_score(grit_roll, csr_roll, sqn_roll)
    stat_sheet_dict = {
        # Note: commented out items should be included afterwords
        # 'ticker': symbol,
        # 'tstmt': ticker_stmt,
        # 'st': st,
        # 'mt': mt,
        "perf": cumul_returns_pct,
        "excess": cumul_excess,
        "trades": trade_count,
        "win": win_rate,
        "win_roll": win_rate_roll,
        "avg_win": avg_win,
        "avg_win_roll": avg_win_roll,
        "avg_loss": avg_loss,
        "avg_loss_roll": avg_loss_roll,
        # 'geo_GE': round(geo_ge, 4),
        "expectancy": edge_expanding,
        "edge_roll": edge_roll,
        "grit": grit_expanding,
        "grit_roll": grit_roll,
        "csr": csr_expanding,
        "csr_roll": csr_roll,
        "pr": pr_expanding,
        "pr_roll": pr_roll,
        "tail": tr_expanding,
        "tail_roll": tr_roll,
        "sqn": sqn_expanding,
        "sqn_roll": sqn_roll,
        "risk_adjusted_returns": score_expanding,
        "risk_adj_returns_roll": score_roll,
    }

    historical_stat_sheet = pd.DataFrame.from_dict(stat_sheet_dict)
    # historical_stat_sheet = historical_stat_sheet.ffill()

    return historical_stat_sheet


def win_rate_calc(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    min_periods,
    round_to=2,
):
    """"""
    price_data = round(price_data, round_to)
    signal_table = pda.SignalTable(signals.copy())
    signal_table.data["trade_count"] = signal_table.counts
    signals_un_pivot = signal_table.unpivot()
    signals_un_pivot = signals_un_pivot.loc[
        ~signals_un_pivot.index.duplicated(keep="last")
    ]
    signals_un_pivot = signals_un_pivot[['dir', 'trade_count']]
    signals_un_pivot = expand_index(signals_un_pivot, price_data.index)
    signals_un_pivot.dir = signals_un_pivot.dir.fillna(0)

    passive_returns_1d = pda_utils.simple_log_returns(price_data.close)
    signals_un_pivot["strategy_returns_1d"] = passive_returns_1d * signals_un_pivot.dir
    # don't use entry date to calculate returns
    signals_un_pivot.loc[signal_table.entry, "strategy_returns_1d"] = 0
    strategy_returns_1d = signals_un_pivot.strategy_returns_1d.copy()
    cumul_returns = pda_utils.cumulative_returns_pct(strategy_returns_1d, min_periods)
    # Cumulative t-stat
    win_count = (
        strategy_returns_1d.loc[strategy_returns_1d > 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    total_count = (
        strategy_returns_1d.loc[strategy_returns_1d != 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )
    win_rate = (win_count / total_count).fillna(method="ffill")
    return win_rate


def main_re_download_data(other_json_path, base_json_path):
    """
    re download stock and bench data, write to locations specified in paths.json
    set index to int and store date index in separate series
    """
    sp500_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ticks, _ = scanner.get_wikipedia_stocks(sp500_wiki)
    data_loader = DataLoader.init_from_paths(other_json_path, base_json_path)
    bench = 'SPY'
    days = 59
    interval = 15
    interval_str = f'{interval}m'
    history_path = data_loader.history_path(bench=bench, interval=interval_str)
    bench_path = data_loader.bench_path(bench=bench, interval=interval_str)
    downloaded_data = scanner.yf_download_data(ticks, days, interval_str)
    downloaded_data = downloaded_data.reset_index()
    dd_date_time = downloaded_data[downloaded_data.columns.to_list()[0]]
    bench_data = scanner.yf_get_stock_data('SPY', days, interval_str)
    bench_data = bench_data.reset_index()
    bd_date_time = bench_data[bench_data.columns.to_list()[0]]

    assert dd_date_time.equals(bd_date_time)

    downloaded_data = downloaded_data[downloaded_data.columns.to_list()[1:]]
    bench_data = bench_data[bench_data.columns.to_list()[1:]]

    relative = simple_relative(downloaded_data, bench_data.close)

    # TODO add relative data to schema
    relative.to_csv(f'{data_loader.base}\\spy_relative_history_{interval_str}.csv')
    downloaded_data.to_csv(history_path)
    bench_data.to_csv(bench_path)
    dd_date_time.to_csv(data_loader.file_path(f'date_time_{interval_str}.csv'))


def load_scan_data(ticker_wiki_url, other_path, base_path, days, interval, benchmark_id):
    """load data for scan"""
    ticks, _ = scanner.get_wikipedia_stocks(ticker_wiki_url)
    _interval = interval['num']
    _interval_type = interval['type']
    interval_str = f'{_interval}{_interval_type}'
    data_loader = DataLoader.init_from_paths(other_path, base_path)
    price_data = pd.read_csv(data_loader.history_path(benchmark_id, interval_str), index_col=0, header=[0, 1]).astype('float64')
    price_glob = PriceGlob(price_data).swap_level()
    bench = pd.read_csv(data_loader.bench_path(benchmark_id, interval_str), index_col=0).astype('float64')
    return ticks, price_glob, bench, benchmark_id, interval_str, _interval, data_loader


def scan_inst(
        _ticks: t.List[str],
        price_glob: t.Any,
        bench: pd.DataFrame,
        benchmark_id: str,
        scan_params,
        strategy_simulator,
        expected_exceptions,
) -> scanner.ScanData:
    scan = scanner.StockDataGetter(
        # data_getter_method=lambda s: scanner.yf_get_stock_data(s, days=days, interval=interval_str),
        data_getter_method=lambda s: price_glob.get_prices(s),
    ).yield_strategy_data(
        bench_symbol=benchmark_id,
        symbols=_ticks,
        # symbols=['FAST'],
        strategy=lambda pdf_, _: (
            strategy_simulator(
                price_data=scanner.data_to_relative(pdf_, bench),
                abs_price_data=pdf_,
                **scan_params['strategy_params']
            )
        ),
        expected_exceptions=expected_exceptions
    )
    scan_data = scanner.run_scanner(
        scanner=scan,
        stat_calculator=lambda data_, entry_signals_: calc_stats(
            data_,
            entry_signals_,
            **scan_params['stat_params']
            # freq='1D',
            # freq='5T',
        ),
        restrict_side=scan_params['scanner_settings']['restrict_side']
    )
    return scan_data


def multiprocess_scan(_scanner, scan_args, ticks_list, interval_str, data_loader):
    with Pool(None) as p:
        results: t.List[scanner.ScanData] = p.map(_scanner, [(ticks,) + scan_args for ticks in ticks_list])

    _stats = []
    _entries = []
    _peaks = []
    _strategy_lookup = {}
    for scan_data in results:
        _stats.append(scan_data.stat_overview)
        _strategy_lookup |= scan_data.strategy_lookup
        _entries.append(scan_data.entry_table)
        _peaks.append(scan_data.peak_table)

    _stat_overview = pd.concat(_stats)
    _entries_table = pd.concat(_entries)
    _peak_table = pd.concat(_peaks)
    # stat_overview_ = stat_overview_.sort_values('risk_adj_returns_roll', axis=1, ascending=False)
    _stat_overview.to_csv(data_loader.file_path(f'stat_overview_{interval_str}.csv'))
    pkl_fp = data_loader.file_path('strategy_lookup.pkl')
    entry_fp = data_loader.file_path(f'entry_table_{interval_str}.pkl')
    peak_fp = data_loader.file_path(f'peak_table_{interval_str}.pkl')
    _entries_table.to_pickle(entry_fp)
    _peak_table.to_pickle(peak_fp)
    with open(pkl_fp, 'wb') as f:
        pickle.dump(_strategy_lookup, f)
    print('done')


class PriceGlob:
    """table of price data for multiple tickers, columns multi index is ticker and (open, high, low, close)"""

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def swap_level(self):
        """swap the level of columns, allow for searching all price data by ticker"""
        return self.__class__(self._data.swaplevel(-1, -2, axis=1))

    def relative_rebased(self, bench_close):
        """relative close prices, rebased to 1"""
        bench_close_frame = self._data.copy()
        bench_close_frame.close = bench_close
        return self._data.close.div(bench_close_frame.close * self._data.close.iloc[0]) * bench_close_frame.close.iloc[0]

    def relative_rebase_all(self, bench_close):
        """all relative prices, rebased to 1"""
        bench_close_frame = self._data.copy()
        bench_close_frame.open = bench_close
        bench_close_frame.high = bench_close
        bench_close_frame.low = bench_close
        bench_close_frame.close = bench_close
        return self._data.div(bench_close_frame * self._data.iloc[0]) * bench_close_frame.iloc[0]

    def relative_all(self, bench_close, rebase=True):
        """all relative, rebased to stock price"""
        bench_close_frame = self._data.copy()
        bench_close_frame.open = bench_close
        bench_close_frame.high = bench_close
        bench_close_frame.low = bench_close
        bench_close_frame.close = bench_close
        if rebase is True:
            bench_close_frame = bench_close_frame.div(bench_close_frame.iloc[0])

        return self.__class__(self._data.div(bench_close_frame))

    def get_prices(self, column):
        d = None
        try:
            d = self._data[column]
        except KeyError:
            pass
        return d


def mp_scan_inst(_args):
    return scan_inst(*_args)


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [
        alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
        for i in range(wanted_parts)
    ]


def main(scan_args, strategy_simulator, expected_exceptions):
    scanner_settings = scan_args['scanner_settings']
    multiprocess = scanner_settings['multiprocess']

    (__ticks, __price_glob, __bench,
     __benchmark_id, __interval_str,
     __interval, __data_loader) = load_scan_data(**scan_args['load_data'])
    __price_glob._data = __price_glob.data.ffill()
    list_of_tickers = split_list(__ticks, cpu_count()-1)
    # list_of_tickers = split_list(['OKE', 'CSCO', 'NLOK'], cpu_count()-1)
    start = perf_counter()
    if multiprocess:
        multiprocess_scan(
            mp_scan_inst,
            (__price_glob, __bench, __benchmark_id, scan_args, strategy_simulator, expected_exceptions),
            list_of_tickers, __interval_str, __data_loader
        )
        print(perf_counter()-start)
    else:
        if scanner_settings['test_symbols'] is False:
            _t = __ticks[30:60]
        else:
            _t = scanner_settings['test_symbols']
        scan_res = scan_inst(
            _ticks=_t,
            price_glob=__price_glob,
            bench=__bench,
            benchmark_id=__benchmark_id,
            scan_params=scan_args,
            strategy_simulator=strategy_simulator,
            expected_exceptions=expected_exceptions
        )


if __name__ == '__main__':
    with open('scan_args.json', 'r') as args_fp:
        _args = json.load(args_fp)
    main(_args)






