import numpy as np
import pandas as pd
from src.data_handler.data_handler import DataHandler

class PortfolioMetrics:
    def __init__(self, asset_rets: pd.DataFrame, benchmark_rets: pd.Series, periods_per_year: int = 52, rf_in_percent: bool = False):
        self.d = DataHandler()
        common_index = asset_rets.index.intersection(benchmark_rets.index)
        self.asset_rets = asset_rets.loc[common_index].copy()
        self.benchmark_rets = benchmark_rets.loc[common_index].copy()

        if self.asset_rets.empty:
            raise ValueError("No overlapping observations between assets and benchmark.")

        self.periods_per_year = periods_per_year
        self.rf_periodic = self.d.rf / 100 if rf_in_percent else self.d.rf

        self.mu = self.asset_rets.mean().values
        self.cov = self.asset_rets.cov().values
        self.bench = self.benchmark_rets.values
        self.bench_mean = float(np.mean(self.bench))
        self.bench_var = float(np.var(self.bench, ddof=1))

        if self.bench_var <= 1e-12:
            raise ValueError("Benchmark variance is too close to zero.")
    
    def portfolio_return(self, w, annualize):
        rp = float(w @ self.mu)
        return rp * self.periods_per_year if annualize else rp

    def portfolio_std(self, w, annualize):
        vol = float(np.sqrt(w @ self.cov @ w))
        return vol * np.sqrt(self.periods_per_year) if annualize else vol

    def portfolio_path(self, w):
        return self.asset_rets.values @ w

    def portfolio_beta(self, w):
        rp = self.asset_rets.values @ w
        return float(np.cov(rp, self.benchmark_rets.values, ddof=1)[0, 1] / self.bench_var)

    def sharpe_ratio(self, w, annualize):
        port_ret = self.portfolio_return(w, annualize)
        port_vol = self.portfolio_std(w, annualize)

        if port_vol <= 1e-12:
            return np.nan

        sharpe = (port_ret - self.rf_periodic.mean()) / port_vol

        if annualize:
            return float(sharpe * np.sqrt(self.periods_per_year))
        return float(sharpe)

    def implied_alpha(self, w, annualize):
        # Jensen's alpha in the same return frequency as inputs unless annualize=True.

        port_ret = self.portfolio_return(w, annualize)
        beta = self.portfolio_beta(w)

        alpha = port_ret - self.rf_periodic.mean() - beta * (self.bench_mean - self.rf_periodic.mean())

        if annualize:
            return float(alpha * self.periods_per_year)
        return float(alpha)

    def benchmark_return(self, annualize):
        return self.bench_mean * self.periods_per_year if annualize else self.bench_mean
    
    def summary(self, w: np.ndarray, objective_value: float | None = None, annualize: bool = True) -> dict:
        summary = {
            "Expected Return of the Portfolio": self.portfolio_return(w, annualize=annualize),
            "Expected Return of the Benchmark (S&P 500 Index)": self.benchmark_return(annualize=annualize),
            "Volatility": self.portfolio_std(w, annualize=annualize),
            "Sharpe Ratio": self.sharpe_ratio(w, annualize=annualize),
            "Beta": self.portfolio_beta(w),
            "Alpha vs benchmark (S&P 500 Index)": self.implied_alpha(w, annualize=annualize),
        }

        if objective_value is not None:
            summary["Objective Value"] = float(objective_value)

        return summary