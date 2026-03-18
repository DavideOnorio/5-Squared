import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, backtest):
        self.port = backtest.port_return.dropna()
        self.bench = backtest.index.loc[backtest.start_date:].dropna()

        common = self.port.index.intersection(self.bench.index)
        self.port = self.port.loc[common]
        self.bench = self.bench.loc[common]

        self.K = 52

    def annualized_return(self, r):
        return r.mean() * self.K

    def annualized_vol(self, r):
        return r.std() * np.sqrt(self.K)

    def downside_vol(self, r):
        downside = r[r < self.rf]
        return downside.std() * np.sqrt(self.K)

    def max_drawdown(self, r):
        wealth = (1 + r).cumprod()
        peak = wealth.cummax()
        return ((wealth - peak) / peak).min()

    def beta(self):
        cov = np.cov(self.port, self.bench, ddof=1)
        return cov[0, 1] / cov[1, 1]

    def summary(self):
        fmt = lambda x: round(x, 4)

        data = {
            "Annualized Return": [fmt(self.annualized_return(self.port)), fmt(self.annualized_return(self.bench))],
            "Annualized Volatility": [fmt(self.annualized_vol(self.port)), fmt(self.annualized_vol(self.bench))],
            "Max Drawdown": [fmt(self.max_drawdown(self.port)), fmt(self.max_drawdown(self.bench))],
            "Beta": [fmt(self.beta()), "1.0000"],
        }

        return pd.DataFrame(data, index=["Portfolio", "S&P 500"]).T