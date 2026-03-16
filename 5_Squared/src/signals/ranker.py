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
        
        self.score = self.signed_l3_composite(self.full_df).sort_values(ascending=False)

    def signed_l3_composite(self, df: pd.DataFrame) -> pd.Series:

        cubed_sum = np.sum(np.power(df, 3), axis=1)
        final_scores = np.sign(cubed_sum) * np.power(np.abs(cubed_sum), 1/3)

        return pd.Series(final_scores, index=df.index, name="directional_l3")


    


        



    


