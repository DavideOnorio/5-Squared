from src.data_handler.data_handler import DataHandler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from src.signals.ranker import Ranker
import pandas as pd
import numpy as np

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

