from src.data_handler.data_handler import DataHandler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import leaves_list
from src.signals.ranker import Ranker
from src.optimization.portfolio_metrics import PortfolioMetrics
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Get_Weights:
    def __init__(self):
        self.df = DataHandler()
        self.r = Ranker()

        self.tickers = self.r.score.index.tolist()
        
        self.returns = self.df.all_log_returns[-52:][self.tickers].dropna(axis=1)

        self.corr = self.returns.corr()
        
        self.scores = self.r.score
        
        self.weights = self._hrp()
<<<<<<< HEAD
        self.beta_penalized_weights, self.beta_penalized_summary = self.opt_sharpe_beta()
        self.alpha_opt_weights, self.alpha_opt_summary = self.opt_alpha()
        self.alpha_from_hrp_weights, self.alpha_from_hrp_summary = self.opt_alpha_from_hrp()
=======
        
        """weights, summary = self.opt_sharpe_beta()
        self.opt_weights = weights
        self.opt_summary = summary"""
>>>>>>> 381b2d54a8710959f641b4407a2d9b1ef2dc459f

    def _hrp(self):
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(50).index.unique().tolist()

        adj_corr = self.corr.loc[tickers, tickers].dropna(axis=0).dropna(axis=1)
        adj_corr = adj_corr.loc[~adj_corr.index.duplicated(), ~adj_corr.columns.duplicated()]

        tickers  = adj_corr.columns.tolist()
        adj_cov  = self.returns[tickers].cov()

        # Cluster
        dist = np.sqrt((1 - adj_corr) / 2)
        link = linkage(squareform(dist.values, checks=False), method='ward')

        # Sort by dendrogram
        order = leaves_list(link)
        tickers = [tickers[i] for i in order]

        # Recursive bisection
        w = pd.Series(1.0, index=tickers)
        clusters = [tickers]
        while clusters:
            clusters = [c[i:j] for c in clusters for i, j in [(0, len(c)//2), (len(c)//2, len(c))] if len(c) > 1]
            for i in range(0, len(clusters), 2):
                l, r = clusters[i], clusters[i+1]
                def var(items):
                    sub = adj_cov.loc[items, items]
                    iw  = 1 / np.diag(sub.values); iw /= iw.sum()
                    return float(iw @ sub.values @ iw)
                v_l, v_r = var(l), var(r)
                w[l] *= 1 - v_l / (v_l + v_r)
                w[r] *= v_l / (v_l + v_r)

        # Score tilt + 5% cap
        w = w * (self.scores[tickers] / self.scores[tickers].sum())
        w = w / w.sum()
        w = w.clip(upper=0.05)
        return (w / w.sum()).sort_values(ascending=False).round(4)


<<<<<<< HEAD
    def opt_sharpe_beta(self, rf = 0.02, beta_penalty = 0.05, max_weight = 0.1, annualize = False):
=======
    """def opt_sharpe_beta(self, rf = 0.02, beta_penalty = 0.05, max_weight = 0.1, period = 'Annual', annualize = True):
>>>>>>> 381b2d54a8710959f641b4407a2d9b1ef2dc459f
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(100).index.unique().tolist()

        # Maximize: Sharpe ratio - beta_penalty * beta

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.df.r_index).dropna().copy()

        common_index = asset_rets.index.intersection(benchmark_returns.index)
        asset_rets = asset_rets.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        n = len(tickers)
        metrics = PortfolioMetrics(asset_rets=asset_rets, benchmark_rets=benchmark_returns, rf_annual=rf, periods_per_year=52)

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

<<<<<<< HEAD
        return w_opt, summary

            
    def opt_alpha( self, rf: float = 0.02, max_weight: float = 0.05, top_n: int = 50, risk_penalty: float = 0.0, annualize: bool = False):
        """
        Pure alpha optimizer:
            max alpha_p - risk_penalty * variance_p

        If risk_penalty = 0, this is pure alpha maximization.
        """
        valid = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores.loc[valid].sort_values(ascending=False).head(top_n).index.unique().tolist()

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.df.r_index).dropna().copy()

        metrics = PortfolioMetrics(asset_rets=asset_rets, benchmark_rets=benchmark_returns, rf_annual=rf, periods_per_year=52)

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

        w_opt = pd.Series(res.x, index=asset_rets.columns, name="weight")
        w_opt = w_opt[w_opt > 1e-8].sort_values(ascending=False)

        summary = metrics.summary(
            res.x,
            objective_value=-objective(res.x),
            annualize=annualize,
        )
        summary["Number of Holdings"] = int((w_opt > 1e-8).sum())

        self.alpha_opt_weights = w_opt
        self.alpha_opt_summary = summary

        return w_opt, summary

    def opt_alpha_from_hrp(self, rf: float = 0.02, max_weight: float = 0.05,top_n: int = 50, risk_penalty: float = 0.0, hrp_penalty: float = 0.05,
        annualize: bool = False):
        """
        HRP-anchored alpha optimizer:
            max alpha_p - risk_penalty * variance_p - hrp_penalty * ||w - w_hrp||^2
        """
        w_hrp = self._hrp()

        tickers = w_hrp.index.tolist()

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.df.r_index).dropna().copy()

        metrics = PortfolioMetrics(asset_rets=asset_rets, benchmark_rets=benchmark_returns, rf_annual=rf, periods_per_year=52)

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

        w_opt = pd.Series(res.x, index=asset_rets.columns, name="weight")
        w_opt = w_opt[w_opt > 1e-8].sort_values(ascending=False)

        summary = metrics.summary(
            res.x,
            objective_value=-objective(res.x),
            annualize=annualize,
        )
        summary["HRP Distance"] = float(np.sum((res.x - w_hrp_vec) ** 2))
        summary["Number of Holdings"] = int((w_opt > 1e-8).sum())

        self.alpha_from_hrp_weights = w_opt
        self.alpha_from_hrp_summary = summary
        self.hrp_anchor_weights = w_hrp

        return w_opt, summary
    
=======
        return w_opt, summary"""
>>>>>>> 381b2d54a8710959f641b4407a2d9b1ef2dc459f
