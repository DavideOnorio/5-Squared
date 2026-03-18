from src.data_handler.data_handler import DataHandler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from src.signals.ranker import Ranker
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Get_Weights:
    def __init__(self):
        self.df = DataHandler()
        self.r = Ranker()
        self.tickers = self.r.score.index.tolist()
        self.returns = self.df.all_log_returns[-52:][self.tickers]
        self.corr = self.returns.corr()
        self.corr = self.corr.loc[~self.corr.index.duplicated(), ~self.corr.columns.duplicated()]
        self.scores = self.r.score
        self.weights = self._hrp()
        #weights, summary = self.opt_treynor_beta()
        #self.opt_weights = weights
        #self.opt_summary = summary

    def _hrp(self):
        
        valid    = [t for t in self.scores.index if t in self.corr.columns]
        # We define our target tickers here
        tickers = self.scores[valid].sort_values(ascending=False).head(50).index.unique().tolist()

        # Here we start to Cluster
        # We convert the correlation into a distance, since the correlation is [-1 , 1] we need to have a proper distance [0, 1]
        # We MUST slice both matrices and the ticker list to be identical in length
        adj_corr = self.corr.loc[tickers, tickers].copy()
        adj_cov  = self.returns[tickers].cov().copy()
        n_tickers = len(tickers) 

        dist = np.sqrt((1 - adj_corr) / 2)
        # we converts the N×N distance matrix into a 1D array, then builds the dendrogram using Ward's criterion 
        # Crucial: we pass dist.values to ensure no index confusion from pandas
        link = linkage(squareform(dist.values, checks=False), method='ward')

        def quasi_diag(link_mat, n):
            link_mat = link_mat.astype(int)
            sort_ix = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
            while sort_ix.max() >= n:
                sort_ix.index = range(0, len(sort_ix) * 2, 2)
                temp          = sort_ix[sort_ix >= n]
                
                res_i = temp.index
                res_v = temp.values - n # If n is 50, and max is 100, this maps to row 50
                
                sort_ix[res_i] = link_mat[res_v, 0]
                sort_ix = pd.concat([sort_ix, pd.Series(link_mat[res_v, 1], index=res_i + 1)])
                sort_ix = sort_ix.sort_index().reset_index(drop=True)
            return [tickers[i] for i in sort_ix.tolist()]
        
        # Recursive bisection
        ordered  = quasi_diag(link, n_tickers)
        w        = pd.Series(1.0, index=ordered)
        clusters = [ordered]

        while clusters:
            clusters = [c[i:j] for c in clusters
                        for i, j in [(0, len(c)//2), (len(c)//2, len(c))]
                        if len(c) > 1]
            for i in range(0, len(clusters), 2):
                l, r = clusters[i], clusters[i+1]
                def var(items):
                    sub = adj_cov.loc[items, items]
                    iw  = 1 / np.diag(sub.values)
                    iw /= iw.sum()
                    return float(iw @ sub.values @ iw)
                
                v_l, v_r = var(l), var(r)
                alloc = 1 - v_l / (v_l + v_r)
                w[l] *= alloc
                w[r] *= (1 - alloc)

        # Tilt by score
        w = w * (self.scores[tickers] / self.scores[tickers].sum())
        return (w / w.sum()).sort_values(ascending=False).round(2)


    """def opt_treynor_beta(self, rf = 0.02, beta_penalty = 0.05, max_weight = 0.10, period = 'Annual', annualize = True):
        valid   = [t for t in self.scores.index if t in self.corr.columns]
        tickers = self.scores[valid].sort_values(ascending=False).head(50).index.unique().tolist()

        # Maximize: Sharpe ratio - beta_penalty * beta

        asset_rets = self.returns[tickers].dropna().copy()
        benchmark_returns = pd.Series(self.df.SPY).dropna().copy()

        common_index = asset_rets.index.intersection(benchmark_returns.index)
        asset_rets = asset_rets.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        n = len(tickers)
        mu = asset_rets.mean().values
        cov = asset_rets.cov().values
        bench_var = np.var(benchmark_returns.values, ddof=1)

        if bench_var <= 0.05:
            raise ValueError('Benchmark variance is too close to zero.')

        if period == 'Annual':
            rf = rf * 52
        elif period == 'Monthly':
            rf = rf * (52 / 12)
        elif period == 'Weekly':
            rf = rf
        else:
            raise ValueError("Invalid period. Choose 'Annual', 'Monthly', or 'Weekly'.")

        def portfolio_return(w):
            return float(w @ mu)

        def portfolio_std(w):
            return float(np.sqrt(w @ cov @ w))

        def portfolio_beta(w):
            rp = asset_rets.values @ w
            return float(np.cov(rp, benchmark_returns.values, ddof=1)[0, 1] / bench_var)

        def objective(w):
            sigma_p = portfolio_std(w)
            if sigma_p <= 1e-12:
                return 1e10
            treynor = (portfolio_return(w) - rf) / sigma_p
            beta = portfolio_beta(w)
            return (treynor - beta_penalty * beta)

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
            raise ValueError(f"Optimization failed: {res.message}")

        w_opt = pd.Series(res.x, index=tickers, name="weight")
        w_opt = w_opt[w_opt > 1e-8].sort_values(ascending=False)

        rp = asset_rets.values @ res.x
        beta = np.cov(rp, benchmark_returns.values, ddof=1)[0, 1] / bench_var

        if annualize:
            port_ret = float(np.mean(rp) * 52)
            port_vol = float(np.std(rp, ddof=1) * np.sqrt(52))
            bench_ret = float(np.mean(benchmark_returns.values) * 52)
            alpha = (np.mean(rp) - rf / 52 - beta * (np.mean(benchmark_returns.values) - rf / 52)) * 52
            treynor = (port_ret - rf) / port_vol
        else:
            port_ret = float(np.mean(rp))
            port_vol = float(np.std(rp, ddof=1))
            bench_ret = float(np.mean(benchmark_returns.values))
            alpha = np.mean(rp) - rf - beta * (np.mean(benchmark_returns.values) - rf)
            treynor = (port_ret - rf) / port_vol

        summary = {
            'Expected Return of the Portfolio': port_ret,
            'Expected Return of the Benchmark (S&P 500 Index)': bench_ret,
            'Volatility': port_vol,
            'Treynor Ratio': float(treynor),
            'Beta': float(beta),
            'Alpha vs benchmark (S&P 500 Index)': float(alpha),
            'Objective Value': float(treynor - beta_penalty * beta),
            'Number of Holdings': int((w_opt > 1e-8).sum())
        }

        self.beta_penalized_weights = w_opt
        self.beta_penalized_summary = summary

        return w_opt, summary

"""