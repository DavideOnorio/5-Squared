import pandas as pd
from src.data_handler.data_handler import DataHandler

#we make this skip to compute momentum in the "Fama"'s way, we should write something about this better
class Momentum:
    def __init__(self):
        self.handler = DataHandler()
        self.lookback = 252
        self.skip = 21
    
    def momentum_factor(self) -> pd.DataFrame:

        log_returns = self.handler.all_log_returns.shift(1)
        full_window_sum = log_returns.rolling(window=self.lookback).sum()
        skip_window_sum = log_returns.rolling(window=self.skip).sum()
        mom_factor = full_window_sum - skip_window_sum

        return mom_factor.iloc[-1]
