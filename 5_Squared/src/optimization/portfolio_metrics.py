import numpy as np
import pandas as pd
from src.data_handler.data_handler import DataHandler

class PortfolioMetrics:
    def __init__(
        self,
        asset_rets: pd.DataFrame,
        benchmark_rets: pd.Series,
        periods_per_year: int = 52,
        rf_series: pd.Series | None = None,
        rolling_betas: pd.DataFrame | None = None,
        rf_in_percent: bool | str = "auto",
    ):
        common_index = asset_rets.index.intersection(benchmark_rets.index)
        self.asset_rets = asset_rets.loc[common_index].copy()
        self.benchmark_rets = pd.Series(benchmark_rets).loc[common_index].copy()

        if self.asset_rets.empty:
            raise ValueError("No overlapping observations between assets and benchmark.")

        self.periods_per_year = periods_per_year

        if rf_series is None:
            rf_series = DataHandler().rf

        self.rf_series = self._prepare_rf_series(
            rf_series=pd.Series(rf_series).reindex(self.asset_rets.index).ffill(),
            periods_per_year=periods_per_year,
            rf_in_percent=rf_in_percent,
        )

        if self.rf_series.dropna().empty:
            raise ValueError("Risk-free series is empty after alignment.")

        # Use the rf observed at the rebalance date / last available date in the window
        self.rf_current = float(self.rf_series.ffill().iloc[-1])

        self.rolling_betas = None
        self.latest_asset_betas = None

        if rolling_betas is not None:
            rb = rolling_betas.loc[:, ~rolling_betas.columns.duplicated(keep="first")].copy()
            rb = rb.reindex(index=self.asset_rets.index, columns=self.asset_rets.columns).ffill()

            if not rb.empty:
                self.rolling_betas = rb
                last_row = rb.iloc[-1]

                # Only use rolling betas if all selected assets have a beta on the rebalance date
                if last_row.notna().all():
                    self.latest_asset_betas = last_row.astype(float).values

        self.mu = self.asset_rets.mean().values
        self.cov = self.asset_rets.cov().values
        self.bench = self.benchmark_rets.values
        self.bench_mean = float(np.mean(self.bench))
        self.bench_var = float(np.var(self.bench, ddof=1))

        if self.bench_var <= 1e-12:
            raise ValueError("Benchmark variance is too close to zero.")

    @staticmethod
    def _prepare_rf_series(
        rf_series: pd.Series,
        periods_per_year: int,
        rf_in_percent: bool | str = "auto",
    ) -> pd.Series:
        rf = pd.to_numeric(pd.Series(rf_series), errors="coerce").ffill()

        # Auto-detect if series looks like 4.25 instead of 0.0425
        if rf_in_percent == "auto":
            median_abs = rf.abs().median(skipna=True)
            if pd.notna(median_abs) and median_abs > 1:
                rf = rf / 100.0
        elif rf_in_percent:
            rf = rf / 100.0

        # Convert annual yield to periodic log-equivalent
        rf = rf.clip(lower=-0.999999)
        return np.log1p(rf) / periods_per_year

    def portfolio_return(self, w, annualize):
        rp = float(np.asarray(w) @ self.mu)
        return rp * self.periods_per_year if annualize else rp

    def portfolio_std(self, w, annualize):
        vol = float(np.sqrt(np.asarray(w) @ self.cov @ np.asarray(w)))
        return vol * np.sqrt(self.periods_per_year) if annualize else vol

    def portfolio_path(self, w):
        return pd.Series(
            self.asset_rets.values @ np.asarray(w),
            index=self.asset_rets.index,
            name="portfolio_return",
        )

    def portfolio_beta(self, w):
        w = np.asarray(w)

        # Preferred: rolling beta at the current rebalance date
        if self.latest_asset_betas is not None:
            return float(w @ self.latest_asset_betas)

        # Fallback: sample beta over the estimation window
        rp = self.asset_rets.values @ w
        return float(np.cov(rp, self.benchmark_rets.values, ddof=1)[0, 1] / self.bench_var)

    def sharpe_ratio(self, w, annualize):
        rp = self.portfolio_path(w)
        vol = float(rp.std(ddof=1))

        if vol <= 1e-12:
            return np.nan

        sharpe = float((rp.mean() - self.rf_current) / vol)
        return sharpe * np.sqrt(self.periods_per_year) if annualize else sharpe

    def implied_alpha(self, w, annualize):
        # Uses:
        # - mean portfolio return over the lookback window
        # - mean benchmark return over the lookback window
        # - rf at the rebalance date
        # - rolling beta at the rebalance date
        port_ret = float(self.portfolio_path(w).mean())
        beta = self.portfolio_beta(w)

        alpha = port_ret - self.rf_current - beta * (self.bench_mean - self.rf_current)

        return float(alpha * self.periods_per_year) if annualize else float(alpha)

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
    
    def from_backtest(cls, bt, periods_per_year: int = 52):
        if bt.port_return is None:
            raise ValueError("Run the backtest first: bt.run()")

        d = DataHandler()

        common_index = bt.port_return.index.intersection(bt.index.index)

        asset_rets = bt.port_return.loc[common_index].to_frame(name="portfolio")
        benchmark_rets = bt.index.loc[common_index]
        rf_series = d.rf.loc[common_index]

        rolling_betas = DataHandler._compute_rolling_beta(
            asset_returns=asset_rets,
            benchmark_returns=benchmark_rets,
            window=52,
        )

        return cls(
            asset_rets=asset_rets,
            benchmark_rets=benchmark_rets,
            periods_per_year=periods_per_year,
            rf_series=rf_series,
            rolling_betas=rolling_betas,
            rf_in_percent="auto",
        )