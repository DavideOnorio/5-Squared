import pandas as pd

class Momentum:
    def __init__(self, log_returns: pd.DataFrame, lookback: int = 52, skip: int = 4):
        self.log_returns = log_returns
        self.lookback = lookback
        self.skip = skip

    def momentum_factor(self) -> pd.Series:
        shifted = self.log_returns.shift(1)
        full_window_sum = shifted.rolling(window=self.lookback).sum()
        skip_window_sum = shifted.rolling(window=self.skip).sum()
        mom_factor = full_window_sum - skip_window_sum
        return mom_factor.iloc[-1]