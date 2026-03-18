import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, asset_rets: pd.DataFrame, benchmark_rets: pd.Series, rf_annual: float = 0.02, periods_per_year: int = 52):
        common_index = asset_rets.index.intersection(benchmark_rets.index)
        self.asset_rets = asset_rets.loc[common_index].copy()
        self.benchmark_rets = benchmark_rets.loc[common_index].copy()

        if self.asset_rets.empty:
            raise ValueError("No overlapping observations between assets and benchmark.")

        self.periods_per_year = periods_per_year
        self.rf_annual = rf_annual
        self.rf_periodic = rf_annual / periods_per_year

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
        rp = self.portfolio_path(w)
        return float(np.cov(rp, self.bench, ddof=1)[0, 1] / self.bench_var)

    def sharpe_ratio(self, w, annualize):
        port_ret = self.portfolio_return(w, annualize)
        port_vol = self.portfolio_std(w, annualize)

        if port_vol <= 1e-12:
            return np.nan

        sharpe = (port_ret - self.rf_periodic) / port_vol

        if annualize:
            return float(sharpe * np.sqrt(self.periods_per_year))
        return float(sharpe)

    def implied_alpha(self, w, annualize):
        # Jensen's alpha in the same return frequency as inputs unless annualize=True.

        port_ret = self.portfolio_return(w, annualize)
        beta = self.portfolio_beta(w)

        alpha = port_ret - self.rf_periodic - beta * (self.bench_mean - self.rf_periodic)

        if annualize:
            return float(alpha * self.periods_per_year)
        return float(alpha)

    def benchmark_return(self, annualize):
        return self.bench_mean * self.periods_per_year if annualize else self.bench_mean

    '''def summary(self, w, beta_penalty, annualize):
        sharpe = self.sharpe_ratio(w, annualize=annualize)
        beta = self.portfolio_beta(w)

        return {
            "Return of the Portfolio": self.portfolio_return(w, annualize=annualize),
            "Return of the Benchmark (S&P 500 Index)": self.benchmark_return(annualize=annualize),
            "Volatility": self.portfolio_std(w, annualize=annualize),
            "Sharpe Ratio": float(sharpe),
            "Beta": float(beta),
            "Implied Excess Returns vs benchmark (S&P 500 Index)": self.implied_alpha(w, annualize=annualize),
            "Objective Value": float(-self.sharpe_ratio(w, annualize=annualize) + beta_penalty * beta),
        }'''
    
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