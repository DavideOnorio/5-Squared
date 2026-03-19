from src.signals.momentum import Momentum
from src.signals.ranker import Ranker
from src.optimization.get_weights import GetWeights
from src.data_handler.data_handler import DataHandler
import pandas as pd
import numpy as np


class Backtest:
    def __init__(self, start_date: str = '2022-08-25'):
        self.d = DataHandler()

        self.df = self.d.all_closes
        self.fundamental = self.d.fundamental
        self.df_r = self.d.all_log_returns
        self.full_tickers = self.d.ticker_list
        self.index = self.d.r_index

        self.start_date = pd.Timestamp(start_date)

        self.port_return: pd.Series | None = None
        self.cum_portfolio: pd.Series | None = None
        self.cum_index: pd.Series | None = None

    def _filter_tickers(self, backtest_df: pd.DataFrame, year: int) -> pd.DataFrame:
        if year not in self.full_tickers.columns:
            year = max(self.full_tickers.columns)
        tickers = self.full_tickers[year].dropna().tolist()
        valid_tickers = [t for t in tickers if t in backtest_df.columns]
        return backtest_df[valid_tickers]

    def _is_delisted(self, backtest_df: pd.DataFrame) -> list[str]:
        nan_counts = backtest_df.tail(6).isna().sum()
        return nan_counts[nan_counts >= 2].index.tolist()

    def _get_active_tickers(self, date: pd.Timestamp) -> list[str]:
        year = date.year
        if year not in self.full_tickers.columns:
            year = max(self.full_tickers.columns)
        tickers = self.full_tickers[year].dropna().tolist()
        return [t for t in tickers if t in self.df_r.columns]

    def _fill_delisted_returns(self, df_returns: pd.DataFrame) -> pd.DataFrame:
        filled = df_returns.copy()
        for col in filled.columns:
            last_valid = filled[col].last_valid_index()
            if last_valid and last_valid < filled.index[-1]:
                filled.loc[last_valid:, col] = filled.loc[last_valid:, col].fillna(0)
        return filled

    def _build_backtest_df(self, rebal_date: pd.Timestamp) -> pd.DataFrame:
        backtest_df = self.df.copy().truncate(after=rebal_date)
        backtest_df = self._filter_tickers(backtest_df, rebal_date.year)
        delisted = self._is_delisted(backtest_df)
        return backtest_df.drop(columns=delisted, errors='ignore')

    def _compute_weights(self, backtest_df: pd.DataFrame, fund_backtest: pd.DataFrame) -> pd.Series | None:
        backtest_log_returns = np.log(backtest_df / backtest_df.shift(1))

        mo = Momentum(log_returns=backtest_log_returns)
        mom = mo.momentum_factor()
        r = Ranker(fundamental=fund_backtest, momentum=mom)
        w = GetWeights(
            log_returns=backtest_log_returns,
            scores=r.score,
            benchmark_rets=self.d.r_index,
            rf_series=self.d.rf,
            rolling_betas=self.d.beta,
        )
        return w.weights

    def _compute_period_return(self, weights: pd.Series, rebal_date: pd.Timestamp, next_date: pd.Timestamp) -> pd.DataFrame | None:
        active_tickers = self._get_active_tickers(rebal_date)
        df_r_active = self.df_r[active_tickers]
        df_r_active = df_r_active.loc[:, ~df_r_active.columns.duplicated()]
        df_r_active = self._fill_delisted_returns(df_r_active)

        valid_tickers = [t for t in weights.index if t in df_r_active.columns]
        filtered = df_r_active[valid_tickers]
        filtered = filtered[(filtered.index >= rebal_date) & (filtered.index < next_date)]

        if filtered.empty:
            return None

        return filtered.dot(weights[valid_tickers]).to_frame(name="portfolio_return")

    def run(self) -> pd.Series:

        fund_dates = self.fundamental.index.unique().sort_values()
        prior = fund_dates[fund_dates <= self.start_date]
        first_date = prior[-1] if len(prior) > 0 else fund_dates[0]

        rebal_dates = sorted(
            fund_dates[fund_dates >= first_date].tolist()
        )
        all_returns = []

        print("Starting to backtest....")
        for i, rebal_date in enumerate(rebal_dates):
            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.df_r.index[-1]

            backtest_df = self._build_backtest_df(rebal_date)
            fund_backtest = self.fundamental.loc[
                self.fundamental.index == rebal_date,
                self.fundamental.columns.isin(backtest_df.columns)
            ]

            weights = self._compute_weights(backtest_df, fund_backtest)
            period_return = self._compute_period_return(weights, rebal_date, next_date)

            if period_return is not None:
                all_returns.append(period_return)

        portfolio_return = pd.concat(all_returns)
        index_b = self.index.loc[portfolio_return.index[0]:]

        self.port_return = portfolio_return["portfolio_return"]
        self.cum_portfolio = self.port_return.cumsum()
        self.cum_index = index_b.cumsum()

        return self.port_return