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
        self.returns = self.df.all_log_returns[-252:]
        self.corr = self.returns.corr()
        self.scores = self.r.score
        self.weights = self._hrp()

    def _hrp(self):
        
        valid    = [t for t in self.scores.index if t in self.corr.columns]
        tickers  = self.scores[valid].sort_values(ascending=False).head(200).index.tolist()

        # Here we start to Cluster
        # We convert the correlation into a distance, since the correlation is [-1 , 1] we need to have a proper distance [0, 1]
        dist = np.sqrt((1 - self.corr.loc[tickers, tickers]) / 2)
        # we converts the N×N distance matrix into a 1D array, then builds the dendrogram using Ward's criterion 
        link = linkage(squareform(dist.values, checks=False), method='ward')

        def quasi_diag(link, n):
            link    = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            while sort_ix.max() >= n:
                sort_ix.index = range(0, len(sort_ix) * 2, 2)
                temp          = sort_ix[sort_ix >= n]
                sort_ix[temp.index] = link[temp.values - n, 0]
                sort_ix = pd.concat([sort_ix, pd.Series(link[temp.values - n, 1], index=temp.index + 1)])
                sort_ix = sort_ix.sort_index().reset_index(drop=True)
            return dist.columns[sort_ix.tolist()].tolist()
        
        # Recursive bisection
        cov      = self.returns[tickers].cov()
        ordered  = quasi_diag(link, len(tickers))
        w        = pd.Series(1.0, index=ordered)
        clusters = [ordered]

        while clusters:
            clusters = [c[i:j] for c in clusters
                        for i, j in [(0, len(c)//2), (len(c)//2, len(c))]
                        if len(c) > 1]
            for i in range(0, len(clusters), 2):
                l, r = clusters[i], clusters[i+1]
                def var(items):
                    sub = cov.loc[items, items]
                    iw  = 1 / np.diag(sub.values)
                    iw /= iw.sum()
                    return float(iw @ sub.values @ iw)
                alloc = 1 - var(l) / (var(l) + var(r))
                w[l] *= alloc
                w[r] *= (1 - alloc)

        # Tilt by score
        w = w * (self.scores[tickers] / self.scores[tickers].sum())
        return (w / w.sum()).sort_values(ascending=False).round(2)



        

