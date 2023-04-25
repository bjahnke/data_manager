# import src.utils.regime as regime
import data_manager.scanner as scanner
# import src.floor_ceiling_regime as sfcr
import json
import typing as t
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import multiprocessing as mp
from time import perf_counter
import pandas_accessors.utils as pda_utils
import pandas_accessors.accessors as pda
import numpy as np
import sys
from dataclasses import dataclass
import data_manager.data_manager_types as dm_types


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

    @classmethod
    def init_with_config(cls, config: t.Dict):
        dm_config = dm_types.AppConfig.parse_obj(config)
        return cls(dm_config.data_config, dm_config.base)

    @property
    def files(self):
        return self._data.files

    @property
    def base(self):
        return self._base_file_path

    def file_path(self, file_name):
        return fr'{self.base}\{file_name}'

    def history_path(self, *_, **__):
        return self.file_path(self.files['history'])

    def bench_path(self, *_, **__):
        return self.file_path(self.files['bench'])


class YfSourceDataLoader(DataLoader):
    """
    DataLoader class with yfinance integration
    """
    def download_data(
            self,
            stock_table_url: str = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            bench: str = 'SPY',
            days: int = 365,
            interval_str: str = '1d'
    ):
        ticks, _ = scanner.get_wikipedia_stocks(stock_table_url)
        history_path = self.history_path(bench=bench, interval=interval_str)
        bench_path = self.bench_path(bench=bench, interval=interval_str)
        downloaded_data = scanner.yf_download_data(ticks, days, interval_str)
        downloaded_data = downloaded_data.reset_index()
        dd_date_time = downloaded_data[downloaded_data.columns.to_list()[0]]
        bench_data = scanner.yf_get_stock_data(bench, days, interval_str)
        bench_data = bench_data.reset_index()
        bd_date_time = bench_data[bench_data.columns.to_list()[0]]

        assert dd_date_time.equals(bd_date_time)

        downloaded_data = downloaded_data[downloaded_data.columns.to_list()[1:]]
        bench_data = bench_data[bench_data.columns.to_list()[1:]]

        relative = simple_relative(downloaded_data, bench_data.close)

        # TODO add relative data to schema
        relative.to_csv(f'{self.base}\\relative_history.csv')
        downloaded_data.to_csv(history_path)
        bench_data.to_csv(bench_path)
        dd_date_time.to_csv(self.file_path(f'date_time.csv'))

    def load_scan_data(self, ticker_wiki_url, interval, benchmark_id):
        """load data for scan"""
        ticks, _ = scanner.get_wikipedia_stocks(ticker_wiki_url)
        _interval = interval['num']
        _interval_type = interval['type']
        price_data = pd.read_csv(self.history_path(), index_col=0, header=[0, 1]).astype('float64')
        price_glob = PriceGlob(price_data).swap_level()
        bench = pd.read_csv(self.bench_path(), index_col=0).astype('float64')
        return ticks, price_glob, bench, benchmark_id, self

    def multiprocess_scan(self, _scanner, scan_args, ticks_list):
        ticks_list = split_list(ticks_list, mp.cpu_count() - 1)

        def myexcepthook(exctype, value, traceback):
            for p in mp.active_children():
                p.terminate()

        with mp.Pool(None) as p:
            sys.excepthook = myexcepthook
            results: t.List[dm_types.ScanData] = p.map(_scanner, [(ticks,) + scan_args for ticks in ticks_list])

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
        _stat_overview.to_csv(self.file_path(f'stat_overview.csv'))
        pkl_fp = self.file_path('strategy_lookup.pkl')
        entry_fp = self.file_path(f'entry_table.pkl')
        peak_fp = self.file_path(f'peak_table.pkl')
        _entries_table.to_pickle(entry_fp)
        _peak_table.to_pickle(peak_fp)
        with open(pkl_fp, 'wb') as f:
            pickle.dump(_strategy_lookup, f)
        print('done')


def reshape_stock_data(stocks_data):
    # Reset MultiIndex columns
    stocks_data.columns = stocks_data.columns.to_flat_index()

    # Rename columns to 'symbol_attribute' format
    stocks_data.columns = [f'{symbol}_{attribute}' for symbol, attribute in stocks_data.columns]

    # Stack data by stock symbols
    stacked_data = stocks_data.stack(level=0)

    # Reset index and rename columns
    stacked_data = stacked_data.reset_index().rename(columns={'level_0': 'timestamp', 'level_1': 'symbol_attribute'})

    # Split 'symbol_attribute' column into 'symbol' and 'attribute' columns
    stacked_data[['symbol', 'attribute']] = stacked_data['symbol_attribute'].str.split('_', expand=True)

    # Remove 'symbol_attribute' column
    stacked_data = stacked_data.drop(columns=['symbol_attribute'])

    # Pivot table to have separate columns for open, high, low, and close
    final_data = stacked_data.pivot_table(index=['timestamp', 'symbol'], columns='attribute', values=0).reset_index()

    return final_data



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
    if round_to != -1:
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


def main_re_download_data(data_manager_config: t.Dict):
    """
    pull latest stock data and store locally
    :param other_json_path:
    :param base_json_path:
    :return:
    """
    """
    re download stock and bench data, write to locations specified in paths.json
    set index to int and store date index in separate series
    """
    sp500_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ticks, _ = scanner.get_wikipedia_stocks(sp500_wiki)
    data_loader = DataLoader.init_with_config(data_manager_config)
    bench = 'SPY'
    days = 365
    interval_str = f'1d'
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
    relative.to_csv(f'{data_loader.base}\\relative_history.csv')
    downloaded_data.to_csv(history_path)
    bench_data.to_csv(bench_path)
    # need to transpose data to remove multiindex,
    # add symbol column (add as PriceGlob function)
    dd_date_time.to_csv(data_loader.file_path(f'date_time.csv'))


def get_smp_data():
    """
    pull smp data from wiki page
    :return:
    """
    return scanner.get_wikipedia_stocks("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

def load_scan_data(ticker_wiki_url, other_path, base_path, interval, benchmark_id):
    """load data for scan"""
    ticks, _ = scanner.get_wikipedia_stocks(ticker_wiki_url)
    _interval = interval['num']
    _interval_type = interval['type']
    data_loader = DataLoader.init_from_paths(other_path, base_path)
    price_data = pd.read_csv(data_loader.history_path(), index_col=0, header=[0, 1]).astype('float64')
    price_glob = PriceGlob(price_data).swap_level()
    bench = pd.read_csv(data_loader.bench_path(), index_col=0).astype('float64')
    return ticks, price_glob, bench, benchmark_id, data_loader


@dataclass
class ScanData:
    ticks: t.List[str]
    price_glob: pd.DataFrame
    bench: pd.DataFrame
    benchmark_id: str
    data_loader: DataLoader


def scan_inst(
        _ticks: t.List[str],
        price_glob: t.Any,
        bench: t.Union[None, pd.DataFrame],
        benchmark_id: t.Union[None, str],
        scan_params,
        strategy_simulator,
        expected_exceptions,
        capital=None,
        available_capital=None
) -> dm_types.ScanData:
    """
    Scan Instance, workflow which run a strategy simulator and calculates performance statistics on results.
    Creates a yield generator with the strategy simulator and passes that to the scanner
    :param _ticks:
    :param price_glob:
    :param bench:
    :param benchmark_id:
    :param scan_params:
    :param strategy_simulator:
    :param expected_exceptions:
    :param capital:
    :param available_capital:
    :return:
    """
    scan = scanner.StockDataGetter(
        # data_getter_method=lambda s: scanner.yf_get_stock_data(s, days=days, interval=interval_str),
        data_getter_method=lambda s: price_glob.get_prices(s),
    ).yield_strategy_data(
        bench_symbol=benchmark_id,
        symbols=_ticks,
        # symbols=['FAST'],
        strategy=lambda pdf_, _: (
            strategy_simulator(
                price_data=scanner.data_to_relative(pdf_, bench) if bench is not None else pdf_,
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
        restrict_side=scan_params['scanner_settings']['restrict_side'],
        capital=capital,
        available_capital=available_capital
    )
    return scan_data

def mp_regime_analysis(args):
    return regime_analysis(*args)

def regime_analysis(
        _ticks: t.List[str],
        price_glob: t.Any,
        bench: t.Union[None, pd.DataFrame],
        benchmark_id: t.Union[None, str],
        scan_params,
        strategy_simulator,
        expected_exceptions,
) -> t.Dict[str, t.Any]:
    """
    Performs regime analysis on the given symbols

    :param _ticks:
    :param price_glob:
    :param bench:
    :param benchmark_id:
    :param scan_params:
    :param strategy_simulator:
    :param expected_exceptions:
    :param capital:
    :param available_capital:
    :return:
    """
    scan_generator = scanner.StockDataGetter(
        # data_getter_method=lambda s: scanner.yf_get_stock_data(s, days=days, interval=interval_str),
        data_getter_method=lambda s: price_glob.get_prices(s),
    ).yield_strategy_data(
        bench_symbol=benchmark_id,
        symbols=_ticks,
        # symbols=['FAST'],
        strategy=lambda pdf_, _: (
            strategy_simulator(
                price_data=scanner.data_to_relative(pdf_, bench) if bench is not None else pdf_,
                abs_price_data=pdf_,
                **scan_params['strategy_params']
            )
        ),
        expected_exceptions=expected_exceptions
    )
    res = {symbol: strategy_data for symbol, _, __, strategy_data in scan_generator}
    return res


def multiprocess_scan(_scanner, scan_args, ticks_list, data_loader):
    ticks_list = split_list(ticks_list, mp.cpu_count() - 1)
    def myexcepthook(exctype, value, traceback):
        for p in mp.active_children():
            p.terminate()

    with mp.Pool(None) as p:
        sys.excepthook = myexcepthook
        results: t.List[dm_types.ScanData] = p.map(_scanner, [(ticks,) + scan_args for ticks in ticks_list])


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
    _stat_overview.to_csv(data_loader.file_path(f'stat_overview.csv'))
    pkl_fp = data_loader.file_path('strategy_lookup.pkl')
    entry_fp = data_loader.file_path(f'entry_table.pkl')
    peak_fp = data_loader.file_path(f'peak_table.pkl')
    _entries_table.to_pickle(entry_fp)
    _peak_table.to_pickle(peak_fp)
    with open(pkl_fp, 'wb') as f:
        pickle.dump(_strategy_lookup, f)
    print('done')


def mp_analysis(_scanner, scan_args, ticks_list):
    ticks_list = split_list(ticks_list, mp.cpu_count() - 1)
    def myexcepthook(exctype, value, traceback):
        for p in mp.active_children():
            p.terminate()

    with mp.Pool(None) as p:
        sys.excepthook = myexcepthook
        results: t.List[dm_types.ScanData] = p.map(_scanner, [(ticks,) + scan_args for ticks in ticks_list])

    return {k: v for d in results for k, v in d.items()}


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

    def un_glob(self) -> pd.DataFrame:
        # Check the position of the 'symbol' and 'OHLC' levels in the columns MultiIndex
        stock_data = self._data.copy()
        if not isinstance(stocks_data.columns, pd.MultiIndex):
            raise ValueError("The input DataFrame should have a MultiIndex with 'symbol' and 'OHLC' levels.")

        first_element = stocks_data.columns[0][0]
        if isinstance(first_element, str):  # Symbol is in the first level
            symbol_position, ohlc_position = 0, 1
        else:  # OHLC is in the first level
            symbol_position, ohlc_position = 1, 0

        # Reset MultiIndex columns
        stocks_data.columns = stocks_data.columns.to_flat_index()

        # Rename columns to 'symbol_attribute' format
        if symbol_position < ohlc_position:
            stocks_data.columns = [f'{symbol}_{attribute}' for symbol, attribute in stocks_data.columns]
        else:
            stocks_data.columns = [f'{attribute}_{symbol}' for attribute, symbol in stocks_data.columns]

        # Stack data by stock symbols
        stacked_data = stocks_data.stack(level=symbol_position)

        # Reset index and rename columns
        stacked_data = stacked_data.reset_index().rename(
            columns={'level_0': 'timestamp', 'level_1': 'symbol_attribute'})

        # Split 'symbol_attribute' column into 'symbol' and 'attribute' columns
        if symbol_position < ohlc_position:
            stacked_data[['symbol', 'attribute']] = stacked_data['symbol_attribute'].str.split('_', expand=True)
        else:
            stacked_data[['attribute', 'symbol']] = stacked_data['symbol_attribute'].str.split('_', expand=True)

        # Remove 'symbol_attribute' column
        stacked_data = stacked_data.drop(columns=['symbol_attribute'])

        # Pivot table to have separate columns for open, high, low, and close
        final_data = stacked_data.pivot_table(index=['timestamp', 'symbol'], columns='attribute',
                                              values=0).reset_index()

        return final_data


def mp_scan_inst(_args):
    return scan_inst(*_args)


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [
        alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
        for i in range(wanted_parts)
    ]


def main(scan_args, strategy_simulator, expected_exceptions, scan_data, capital=None, available_capital=None):
    scanner_settings = scan_args['scanner_settings']
    multiprocess = scanner_settings['multiprocess']

    (__ticks, __price_glob, __bench,
     __benchmark_id, __data_loader) = scan_data  # load_scan_data(**scan_args['load_data'])
    __price_glob._data = __price_glob.data.ffill()
    # list_of_tickers = split_list(['OKE', 'CSCO', 'NLOK'], cpu_count()-1)
    start = perf_counter()
    if multiprocess:
        multiprocess_scan(
            mp_scan_inst,
            (
                __price_glob,
                __bench,
                __benchmark_id,
                scan_args,
                strategy_simulator,
                expected_exceptions,
                capital,
                available_capital
            ),
            __ticks,
            __data_loader
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
            expected_exceptions=expected_exceptions,
            capital=capital,
            available_capital=available_capital
        )


if __name__ == '__main__':
    main_re_download_data()
    with open("C:\\Users\\bjahn\\PycharmProjects\\backtest_notebook\\data_args\\scan_args.json", "r") as args_fp:
        _args = json.load(args_fp)
    main(_args)


