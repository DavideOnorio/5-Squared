import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from src.optimization.portfolio_metrics import PortfolioMetrics
from src.data_handler import DataHandler
from scipy.optimize import minimize

class GetWeights:
    def __init__(self, log_returns: pd.DataFrame, scores: pd.Series, lookback: int = 52, top_n: int = 50, max_weight: float = 0.05):
        self.scores = scores
        self.returns = log_returns[-lookback:].dropna(axis=1)
        self.corr = self.returns.corr()
        self.top_n = top_n
        self.max_weight = max_weight
        self.d = DataHandler()
        self.index = self.d.r_index
        self.rf = self.d.rf
        self.beta = self.d.beta

        #self.weights = self._hrp()

        # Sharpe beta optimzation
        beta_weights, beta_summary = self.opt_sharpe_beta(beta_penalty=0.05, max_weight=max_weight, annualize=True)
        self.weights = beta_weights
        self.summary = beta_summary

        # Alpha optimization
        '''alpha_weights, alpha_summary = self.opt_alpha(max_weight=max_weight, top_n=top_n, risk_penalty=0.0, annualize=True)
        self.weights = alpha_weights
        self.summary = alpha_summary'''

        # Alpha optimization from HRP
        '''alpha_hrp_weights, alpha_hrp_summary = self.opt_alpha_from_hrp(max_weight=max_weight, top_n=top_n, risk_penalty=0.0, hrp_penalty=0.05, annualize=True)
        self.weights = alpha_hrp_weights
        self.summary = alpha_hrp_summary'''


    def _select_tickers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        valid = [t for t in self.scores.index if t in self.corr.columns]
        tickers = (
            self.scores[valid]
            .sort_values(ascending=False)
            .head(self.top_n)
            .index.unique()
            .tolist()
        )

        corr = self.corr.loc[tickers, tickers].dropna(axis=0).dropna(axis=1)
        corr = corr.loc[~corr.index.duplicated(), ~corr.columns.duplicated()]

        tickers = corr.columns.tolist()
        cov = self.returns[tickers].cov()
        return corr, cov

    def _cluster_order(self, corr: pd.DataFrame) -> list[str]:
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist.values, checks=False), method='ward')
        order = leaves_list(link)
        return [corr.columns[i] for i in order]

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
        sub = cov.loc[items, items]
        inv_w = 1 / np.diag(sub.values)
        inv_w /= inv_w.sum()
        return float(inv_w @ sub.values @ inv_w)

    def _recursive_bisection(self, tickers: list[str], cov: pd.DataFrame) -> pd.Series:
        w = pd.Series(1.0, index=tickers)
        clusters = [tickers]
        while clusters:
            clusters = [
                c[i:j]
                for c in clusters
                for i, j in [(0, len(c) // 2), (len(c) // 2, len(c))]
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                left, right = clusters[i], clusters[i + 1]
                v_l = self._cluster_variance(cov, left)
                v_r = self._cluster_variance(cov, right)
                w[left] *= 1 - v_l / (v_l + v_r)
                w[right] *= v_l / (v_l + v_r)
        return w

    def _apply_score_tilt(self, w: pd.Series) -> pd.Series:
        s = self.scores[w.index]
        s = (s - s.min() + 1e-8) ** 0.3
        tilted = w * s
        return tilted / tilted.sum()

    def _cap_weights(self, w: pd.Series) -> pd.Series:
        w = w.copy()
        for _ in range(10):
            breach = w[w > self.max_weight]
            if breach.empty:
                break
            w[breach.index] = self.max_weight
            under = w[w < self.max_weight]
            w[under.index] *= (1 - len(breach) * self.max_weight) / under.sum()
        return w

    def _hrp(self) -> pd.Series:
        corr, cov = self._select_tickers()
        tickers = self._cluster_order(corr)
        cov = cov.loc[tickers, tickers]

        w = self._recursive_bisection(tickers, cov)
        w = self._apply_score_tilt(w)
        w = self._cap_weights(w)

        return (w / w.sum()).sort_values(ascending=False).round(4)


    def opt_sharpe_beta(self, beta_penalty = 0.05, max_weight = 0.1, annualize = False):
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(100).index.unique().tolist()

        # Maximize: Sharpe ratio - beta_penalty * beta

        asset_rets = self.returns.loc[:, tickers].copy()
        asset_rets = asset_rets.loc[:, ~asset_rets.columns.duplicated(keep="first")]
        asset_rets = asset_rets.dropna()

        benchmark_returns = pd.Series(self.index).dropna().copy()

        common_index = asset_rets.index.intersection(benchmark_returns.index)
        asset_rets = asset_rets.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        n = asset_rets.shape[1]
        rolling_betas = self.beta.reindex(index=asset_rets.index, columns=asset_rets.columns)
        rf_series = self.rf.reindex(asset_rets.index)

        metrics = PortfolioMetrics(
            asset_rets=asset_rets,
            benchmark_rets=benchmark_returns,
            periods_per_year=52,
            rf_series=rf_series,
            rolling_betas=rolling_betas,
            rf_in_percent="auto",
        )

        if metrics.bench_var <= 1e-12:
            raise ValueError('Benchmark variance is too close to zero.')
        if n * max_weight < 1:
            raise ValueError("Infeasible max_weight: top_n * max_weight must be at least 1.")
        
        def objective(w):
            sigma_p = metrics.portfolio_std(w, annualize)
            if sigma_p <= 1e-12:
                return 1e10
            sharpe = metrics.sharpe_ratio(w, annualize)
            beta = metrics.portfolio_beta(w)
            return -(sharpe - beta_penalty * beta)

        x0 = np.repeat(1/n, n) # first guess is equal weights
        bounds = [(0.0, max_weight)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not res.success:
            raise ValueError(f'Optimization failed: {res.message}')

        w_opt = pd.Series(res.x, index = tickers, name = 'weight')
        w_opt = w_opt.sort_values(ascending=False).round(4)

        summary = metrics.summary(res.x, annualize=annualize)
        summary['Number of Holdings'] = int((w_opt > 1e-8).sum())

        self.beta_penalized_weights = w_opt
        self.beta_penalized_summary = summary

        return w_opt, summary

            
    def opt_alpha( self, max_weight: float = 0.05, top_n: int = 50, risk_penalty: float = 0.0, annualize: bool = False):
        
        '''Pure alpha optimizer:
            max alpha_p - risk_penalty * variance_p

        If risk_penalty = 0, this is pure alpha maximization.'''
    
        valid = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores.loc[valid].sort_values(ascending=False).head(top_n).index.unique().tolist()

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.index).dropna().copy()

        rolling_betas = self.beta.reindex(index=asset_rets.index, columns=asset_rets.columns)
        rf_series = self.rf.reindex(asset_rets.index)

        metrics = PortfolioMetrics(
            asset_rets = asset_rets,
            benchmark_rets = benchmark_returns,
            periods_per_year = 52,
            rf_series = rf_series,
            rolling_betas = rolling_betas,
            rf_in_percent = "auto",
        )

        n = asset_rets.shape[1]

        if metrics.bench_var <= 1e-12:
            raise ValueError('Benchmark variance is too close to zero.')
        if n * max_weight < 1:
            raise ValueError("Infeasible max_weight: top_n * max_weight must be at least 1.")

        def objective(w):
            alpha = metrics.implied_alpha(w, annualize=False)
            variance = float(w @ metrics.cov @ w)
            return -alpha + risk_penalty * variance

        x0 = np.repeat(1 / n, n)
        bounds = [(0.0, max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
        
        w_opt = pd.Series(res.x, index = tickers, name = 'weight')
        w_opt = w_opt.sort_values(ascending=False).round(4)

        summary = metrics.summary(res.x, annualize=annualize)
        summary['Number of Holdings'] = int((w_opt > 1e-8).sum())

        self.alpha_opt_weights = w_opt
        self.alpha_opt_summary = summary

        return w_opt, summary

    def opt_alpha_from_hrp(self, max_weight: float = 0.05,top_n: int = 50, risk_penalty: float = 0.0, hrp_penalty: float = 0.05,
        annualize: bool = False):
        
        '''HRP-anchored alpha optimizer:
            max alpha_p - risk_penalty * variance_p - hrp_penalty * ||w - w_hrp||^2'''
        
        w_hrp = self._hrp()

        tickers = w_hrp.index.tolist()

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.index).dropna().copy()

        rolling_betas = self.beta.reindex(index=asset_rets.index, columns=asset_rets.columns)
        rf_series = self.rf.reindex(asset_rets.index)

        metrics = PortfolioMetrics(
            asset_rets=asset_rets,
            benchmark_rets=benchmark_returns,
            periods_per_year=52,
            rf_series=rf_series,
            rolling_betas=rolling_betas,
            rf_in_percent="auto",
        )

        n = asset_rets.shape[1]
        w_hrp_vec = w_hrp.loc[asset_rets.columns].values

        if n * max_weight < 1 - 1e-12:
            raise ValueError(
                f"Infeasible optimization: {n} assets with max_weight={max_weight:.2%} cannot sum to 100%."
            )

        def objective(w):
            alpha = metrics.implied_alpha(w, annualize=False)
            variance = float(w @ metrics.cov @ w)
            hrp_distance = float(np.sum((w - w_hrp_vec) ** 2))
            return -alpha + risk_penalty * variance + hrp_penalty * hrp_distance

        x0 = w_hrp_vec.copy()
        bounds = [(0.0, max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")

        w_opt = pd.Series(res.x, index = tickers, name = 'weight')
        w_opt = w_opt.sort_values(ascending=False).round(4)

        summary = metrics.summary(res.x, annualize=annualize)
        summary["HRP Distance"] = float(np.sum((res.x - w_hrp_vec) ** 2))
        summary["Number of Holdings"] = int((w_opt > 1e-8).sum())

        self.alpha_from_hrp_weights = w_opt
        self.alpha_from_hrp_summary = summary
        self.hrp_anchor_weights = w_hrp

        return w_opt, summary