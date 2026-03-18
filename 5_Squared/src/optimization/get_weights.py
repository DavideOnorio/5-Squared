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
        
        weights, summary = self.opt_sharpe_beta()
        self.opt_weights = weights
        self.opt_summary = summary

    def _hrp(self):
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(100).index.unique().tolist()

        adj_corr = self.corr.loc[tickers, tickers].dropna(axis=0).dropna(axis=1)
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


    def opt_sharpe_beta(self, rf = 0.02, beta_penalty = 0.05, max_weight = 0.1, period = 'Annual', annualize = False):
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(50).index.unique().tolist()

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

        summary = metrics.summary(res.x, beta_penalty=beta_penalty, annualize=annualize)
        summary['Number of Holdings'] = int((w_opt > 1e-8).sum())

        self.beta_penalized_weights = w_opt
        self.beta_penalized_summary = summary

        return w_opt, summary