import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self):
        self.fundamental = pd.read_excel(r"5_Squared\data\raw\with_roe.xlsx", engine='openpyxl')
        self.fundamental = self.fundamental.set_index('Ticker').rename(columns=lambda x: x.split()[0])

        self.all_closes = pd.read_excel(r"5_Squared\data\raw\sep500_5y.xlsx", engine='openpyxl')
        self.all_closes["Date"] = pd.to_datetime(self.all_closes["Date"], unit="D", origin="1899-12-30")
        self.all_closes = self.all_closes.set_index("Date").rename(columns=lambda x: x.split()[0]).apply(pd.to_numeric, errors="coerce")
        
        self.SPY = self.all_closes['SPX']
        self.all_closes = self.all_closes.drop(columns=['SPX'])
        
        self.all_log_returns = np.log(self.all_closes / self.all_closes.shift(1))

        

    def close(self, ticker) -> pd.Series:
        return self.all_closes[ticker]

    def log_return(self, ticker) -> pd.Series:
        return self.all_log_returns[ticker]

    
    