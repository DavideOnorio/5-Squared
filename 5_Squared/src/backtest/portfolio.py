from src.signals.momentum import Momentum
from src.data_handler.data_handler import DataHandler
from src.signals.ranker import Ranker
from src.optimization.get_weights import Get_Weights
import pandas as pd
import numpy as np

class Backtest:
    def __init__(self):
        self.w = Get_Weights()
        self.d = DataHandler()

        self.df = self.d.all_closes
        self.df_r = self.d.all_log_returns

        self.start_date = pd.Timestamp('2022-04-25')
        self.backtest_df = self.d.all_closes.copy().truncate(after=self.start_date)
        self.delisted = self._is_delisted()
        self.backtest_df = self.backtest_df.drop(columns=self.delisted, errors='ignore')

    def _is_delisted(self):
        nan_counts = self.backtest_df.tail(6).isna().sum()
        delisted_stocks = nan_counts[nan_counts >= 2].index.tolist()
        return delisted_stocks

    def _swap_to_backtest(self):
        self.d.all_closes = self.backtest_df
        self.d.all_log_returns = np.log(self.backtest_df / self.backtest_df.shift(1))

    def _restore(self):
        self.d.all_closes = self.df
        self.d.all_log_returns = self.df_r

    """def run(self):
        self._swap_to_backtest()

        b = DataHandler()
        mo = Momentum()
        r = Ranker()
        w = Get_Weights()

        print(b.all_closes)

        self._restore()"""


    
    

