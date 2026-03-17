import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self):
        self.fundamental = pd.read_excel(r"5_Squared\data\raw\with_roe.xlsx", engine='openpyxl')
        self.fundamental = self.fundamental.set_index('Ticker').rename(columns=lambda x: x.split()[0])

        self.all_closes = pd.read_csv(r"5_Squared\data\raw\data.csv", header=[0, 1], index_col=0)['Close']
        self.all_closes.index = pd.to_datetime(self.all_closes.index)
        self.all_log_returns = np.log(self.all_closes / self.all_closes.shift(1))

    def close(self, ticker) -> pd.Series:
        return self.all_closes[ticker]

    def log_return(self, ticker) -> pd.Series:
        return self.all_log_returns[ticker]

    
    