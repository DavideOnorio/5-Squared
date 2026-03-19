import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, asset_rets: pd.DataFrame, benchmark_rets: pd.Series,
                 rf_series: pd.Series, rolling_betas: pd.DataFrame,
                 periods_per_year: int = 52):

        common = asset_rets.index.intersection(benchmark_rets.index)
        self.asset_rets = asset_rets.loc[common].copy()
        self.benchmark_rets = benchmark_rets.loc[common].copy()
        self.periods_per_year = periods_per_year

        rf = pd.to_numeric(rf_series.reindex(common).ffill(), errors="coerce")
        if rf.abs().median(skipna=True) > 1:
            rf = rf / 100.0
        self.rf_current = float(np.log1p(rf.clip(lower=-0.999999)).iloc[-1] / periods_per_year)

        rb = rolling_betas.reindex(index=common, columns=asset_rets.columns).ffill()
        last_row = rb.iloc[-1] if not rb.empty else pd.Series(dtype=float)
        self.latest_betas = last_row.values if last_row.notna().all() else None

        self.mu = self.asset_rets.mean().values
        self.cov = self.asset_rets.cov().values
        self.bench = self.benchmark_rets.values
        self.bench_mean = float(np.mean(self.bench))
        self.bench_var = float(np.var(self.bench, ddof=1))

    def portfolio_return(self, w, annualize=True):
        rp = float(np.asarray(w) @ self.mu)
        return rp * self.periods_per_year if annualize else rp

    def portfolio_std(self, w, annualize=True):
        w = np.asarray(w)
        vol = float(np.sqrt(w @ self.cov @ w))
        return vol * np.sqrt(self.periods_per_year) if annualize else vol

    def portfolio_beta(self, w):
        w = np.asarray(w)
        if self.latest_betas is not None:
            return float(w @ self.latest_betas)
        rp = self.asset_rets.values @ w
        return float(np.cov(rp, self.bench, ddof=1)[0, 1] / self.bench_var)

    def sharpe_ratio(self, w, annualize=True):
        rp = self.asset_rets.values @ np.asarray(w)
        vol = float(np.std(rp, ddof=1))
        if vol <= 1e-12:
            return np.nan
        sharpe = float((np.mean(rp) - self.rf_current) / vol)
        return sharpe * np.sqrt(self.periods_per_year) if annualize else sharpe

    def implied_alpha(self, w, annualize=True):
        rp = float(np.mean(self.asset_rets.values @ np.asarray(w)))
        beta = self.portfolio_beta(w)
        alpha = rp - self.rf_current - beta * (self.bench_mean - self.rf_current)
        return float(alpha * self.periods_per_year) if annualize else float(alpha)

    def summary(self, w: np.ndarray, annualize: bool = True) -> dict:
        return {
            "Portfolio Return": self.portfolio_return(w, annualize),
            "Benchmark Return": self.bench_mean * self.periods_per_year if annualize else self.bench_mean,
            "Volatility": self.portfolio_std(w, annualize),
            "Sharpe Ratio": self.sharpe_ratio(w, annualize),
            "Beta": self.portfolio_beta(w),
            "Alpha": self.implied_alpha(w, annualize),
        }