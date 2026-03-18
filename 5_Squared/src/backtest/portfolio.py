from src.signals.momentum import Momentum
from src.data_handler.data_handler import DataHandler
from src.signals.ranker import Ranker
from src.optimization.get_weights import Get_Weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self):
        self.d = DataHandler()

        self.df = self.d.all_closes
        self.fundamental = self.d.fundamental
        self.df_r = self.d.all_log_returns
        self.full_tickers = self.d.ticker_list
        self.index = self.d.r_index
        self.cum_portfolio = pd.DataFrame()

        self.start_date = pd.Timestamp('2023-04-25')

    def _filter_tickers(self, backtest_df, year: int):
        tickers = self.full_tickers[year].dropna().tolist()
        valid_tickers = [t for t in tickers if t in backtest_df.columns]
        return backtest_df[valid_tickers]

    def _is_delisted(self, backtest_df):
        nan_counts = backtest_df.tail(6).isna().sum()
        return nan_counts[nan_counts >= 2].index.tolist()

    def _get_active_tickers(self, date: pd.Timestamp) -> list:
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

    def _build_backtest_df(self, rebal_date: pd.Timestamp):
        backtest_df = self.df.copy().truncate(after=rebal_date)
        backtest_df = self._filter_tickers(backtest_df, rebal_date.year)
        delisted = self._is_delisted(backtest_df)
        backtest_df = backtest_df.drop(columns=delisted, errors='ignore')
        return backtest_df

    def _swap_to_backtest(self, backtest_df, fund_backtest):
        self.d.all_closes = backtest_df
        self.d.all_log_returns = np.log(backtest_df / backtest_df.shift(1))
        self.d.fundamental = fund_backtest

    def _restore(self):
        self.d.all_closes = self.df
        self.d.all_log_returns = self.df_r
        self.d.fundamental = self.fundamental

    def run(self):
        
        # Use exactly the dates present in the fundamental file
        rebal_dates = sorted(self.fundamental.index[self.fundamental.index >= self.start_date].unique().tolist())
        all_returns = []
        
        for i, rebal_date in enumerate(rebal_dates):
            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.df_r.index[-1]

            #Truncates everything before the rebal_date
            backtest_df = self._build_backtest_df(rebal_date)
            fund_backtest = self.fundamental.loc[
                self.fundamental.index == rebal_date,  # row: only this quarter's date
                self.fundamental.columns.isin(backtest_df.columns) # columns: only tickers that exist in backtest_df
            ]

            self._swap_to_backtest(backtest_df, fund_backtest)
            mo = Momentum()
            mo.momentum_factor
            r = Ranker()
            w = Get_Weights()
            self._restore()

            active_tickers = self._get_active_tickers(rebal_date)
            df_r_active = self.df_r[active_tickers]
            df_r_active = df_r_active.loc[:, ~df_r_active.columns.duplicated()]
            df_r_active = self._fill_delisted_returns(df_r_active)

            valid_tickers = [t for t in w.weights.index if t in df_r_active.columns]
            filtered = df_r_active[valid_tickers]
            
            # Use >= rebal_date and < next_date to never miss rows
            filtered = filtered[(filtered.index >= rebal_date) & (filtered.index < next_date)]

            if filtered.empty:
                print(f"  No returns data for this period, skipping.")
                continue

            period_return = filtered.dot(w.weights[valid_tickers]).to_frame(name="portfolio_return")
            all_returns.append(period_return)

        portfolio_return = pd.concat(all_returns)
        index_b = self.index.loc[self.start_date:]

        self.port_return = portfolio_return["portfolio_return"]
        self.cum_portfolio = (1 + portfolio_return["portfolio_return"]).cumprod() - 1
        self.cum_index = (1 + index_b).cumprod() - 1
        