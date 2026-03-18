from pathlib import Path
import pandas as pd
import numpy as np

class DataHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        base_path = Path("5_Squared") / "data" / "raw"

        self.fundamental = pd.read_excel(base_path / "with_roe.xlsx", engine='openpyxl')
        self.fundamental = self.fundamental.set_index('Ticker').rename(columns=lambda x: x.split()[0])

        self.all_closes = pd.read_excel(base_path / "sep500_5y.xlsx", engine='openpyxl')
        self.all_closes["Date"] = pd.to_datetime(self.all_closes["Date"], unit="D", origin="1899-12-30")
        self.all_closes = self.all_closes.set_index("Date").rename(columns=lambda x: x.split()[0]).apply(pd.to_numeric, errors="coerce")
        
        self.SPY = self.all_closes['SPX']
        self.all_closes = self.all_closes.drop(columns=['SPX'])

        self.ticker_list = pd.read_excel(r"5_Squared\data\raw\full_stocks_5y.xlsx", engine='openpyxl')
        self.ticker_list = self.ticker_list.map(lambda x: x.split()[0] if isinstance(x, str) else x)

        self.all_log_returns = np.log(self.all_closes / self.all_closes.shift(1))
        self._initialized = True


    def close(self, ticker) -> pd.Series:
        return self.all_closes[ticker]

    def log_return(self, ticker) -> pd.Series:
        return self.all_log_returns[ticker]

    
    