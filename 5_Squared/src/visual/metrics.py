import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, backtest):
        self.port = backtest.port_return.dropna()
        self.bench = backtest.d.r_index.loc[backtest.start_date:].dropna()

        common = self.port.index.intersection(self.bench.index)
        self.port = self.port.loc[common]
        self.bench = self.bench.loc[common]

        self.K = 52

        rf_raw = backtest.d.rf.loc[common].ffill()
        rf_raw = rf_raw / 100 if rf_raw.abs().median() > 1 else rf_raw
        self.rf = np.log1p(rf_raw) / self.K

    def annualized_return(self, r):
        return r.mean() * self.K

    def annualized_vol(self, r):
        return r.std() * np.sqrt(self.K)

    def downside_vol(self, r):
        downside = r[r < self.rf.reindex(r.index).fillna(self.rf.mean())]
        return downside.std() * np.sqrt(self.K)

    def sharpe(self, r):
        rf_mean = self.rf.reindex(r.index).mean()
        return (r.mean() - rf_mean) / r.std() * np.sqrt(self.K)

    def sortino(self, r):
        rf_mean = self.rf.reindex(r.index).mean()
        return (r.mean() - rf_mean) / (self.downside_vol(r) / np.sqrt(self.K)) * np.sqrt(self.K)

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
            "Sharpe Ratio": [fmt(self.sharpe(self.port)), fmt(self.sharpe(self.bench))],
            "Sortino Ratio": [fmt(self.sortino(self.port)), fmt(self.sortino(self.bench))],
            "Max Drawdown": [fmt(self.max_drawdown(self.port)), fmt(self.max_drawdown(self.bench))],
            "Beta": [fmt(self.beta()), "1.0000"],
        }

        return pd.DataFrame(data, index=["Portfolio", "S&P 500"]).T