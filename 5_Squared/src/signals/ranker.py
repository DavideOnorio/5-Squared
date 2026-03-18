import pandas as pd
import numpy as np
from src.data_handler.preprocessing import DataTransformer


class Ranker:
    def __init__(self, fundamental: pd.DataFrame, momentum: pd.Series):
        self.tr = DataTransformer()

        self.full_df = (
            fundamental.T
            .join(momentum.rename('mom'))
            .apply(pd.to_numeric, errors='coerce')
            .pipe(lambda df: df.fillna(df.mean()))
            .apply(self.tr.scale)
        )

        self.score = (
            self.signed_lp_composite(self.full_df)
            .sort_values(ascending=False)
            .pipe(lambda s: s[~s.index.duplicated(keep='first')])
        )

    def signed_lp_composite(self, df: pd.DataFrame, p: float = 0.5) -> pd.Series:
        signed_pow = np.sign(df) * np.power(np.abs(df), p)
        agg = signed_pow.sum(axis=1)
        final_scores = np.sign(agg) * np.power(np.abs(agg), 1 / p)
        return pd.Series(final_scores, index=df.index, name=f"directional_l{p}")