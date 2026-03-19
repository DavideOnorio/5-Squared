import pandas as pd
import numpy as np
from scipy.optimize import minimize
from src.optimization.portfolio_metrics import PortfolioMetrics_for_optimization


class GetWeights:
    def __init__(self, log_returns: pd.DataFrame, scores: pd.Series,
                 benchmark_rets: pd.Series, rf_series: pd.Series,
                 rolling_betas: pd.DataFrame,
                 lookback: int = 52, top_n: int = 100, max_weight: float = 0.06,
                 beta_bonus: float = -0.05, annualize: bool = True):

        self.scores = scores
        self.returns = log_returns[-lookback:].dropna(axis=1)
        self.top_n = top_n
        self.max_weight = max_weight
        self.benchmark_rets = benchmark_rets
        self.rf_series = rf_series
        self.rolling_betas = rolling_betas

        self.weights, self.summary = self._optimize(beta_bonus, annualize)

    def _select_tickers(self) -> list[str] | None:
        valid = [t for t in self.scores.index if t in self.returns.columns]
        if len(valid) < 2:
            return None
        return (
            self.scores[valid]
            .sort_values(ascending=False)
            .head(self.top_n)
            .index.unique()
            .tolist()
        )

    def _build_metrics(self, tickers: list[str]) -> PortfolioMetrics_for_optimization:
        asset_rets = self.returns[tickers].dropna()
        asset_rets = asset_rets.loc[:, ~asset_rets.columns.duplicated(keep="first")]
        common = asset_rets.index.intersection(self.benchmark_rets.index)

        return PortfolioMetrics_for_optimization(
            asset_rets=asset_rets.loc[common],
            benchmark_rets=self.benchmark_rets.loc[common],
            rf_series=self.rf_series.reindex(common),
            rolling_betas=self.rolling_betas.reindex(index=common, columns=asset_rets.columns),
        )

    def _optimize(self, beta_bonus: float, annualize: bool) -> tuple[pd.Series | None, dict | None]:
        tickers = self._select_tickers()
        if tickers is None:
            return None, None

        metrics = self._build_metrics(tickers)
        n = len(tickers)

        def objective(w):
            if metrics.portfolio_std(w, annualize) <= 1e-12:
                return 1e10
            return -(metrics.sharpe_ratio(w, annualize) + beta_bonus * metrics.portfolio_beta(w))

        res = minimize(
            objective,
            x0=np.repeat(1 / n, n),
            method='SLSQP',
            bounds=[(0.0, self.max_weight)] * n,
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
        )

        w_opt = pd.Series(res.x, index=tickers, name='weight')
        w_opt = (w_opt / w_opt.sum()).sort_values(ascending=False).round(4)

        summary = metrics.summary(res.x, annualize=annualize)
        summary['Number of Holdings'] = int((w_opt > 1e-8).sum())

        return w_opt, summary