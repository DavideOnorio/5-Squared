from src.signals.momentum import Momentum
from src.data_handler.preprocessing import DataTransformer
from src.data_handler.data_handler import DataHandler
import pandas as pd
import numpy as np

class Ranker:
    def __init__(self):
        self.mom = Momentum()
        self.tr = DataTransformer()
        self.df = DataHandler()

        self.full_df = (self.df.fundamental.T
                .join(self.mom.momentum_factor().rename('mom'))
                .apply(pd.to_numeric, errors='coerce')
                .pipe(lambda df: df.fillna(df.mean()))
                .apply(self.tr.scale))
        
        self.score = self.signed_lp_composite(self.full_df).sort_values(ascending=False)

    def signed_lp_composite(self, df: pd.DataFrame, p: float = 0.5) -> pd.Series:
        signed_pow = np.sign(df) * np.power(np.abs(df), p)
        agg = signed_pow.sum(axis=1)
        final_scores = np.sign(agg) * np.power(np.abs(agg), 1 / p)
        return pd.Series(final_scores, index=self.full_df.index, name=f"directional_l{p}")


    


        



    


