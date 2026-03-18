from pathlib import Path
import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, base_path: str | Path = Path("5_Squared") / "data" / "raw"):
        base_path = Path(base_path)

        self.fundamental = self._load_fundamental(base_path / "ind_5y.xlsx")
        self.all_closes = self._load_closes(base_path / "sep500_14y.xlsx")
        self.ticker_list = self._load_ticker_list(base_path / "full_stocks_14y.xlsx")

        self.SPY = self.all_closes.pop('SPX')
        self.rf = self.all_closes.pop('USGG10YR')
        self.r_index = np.log(self.SPY / self.SPY.shift(1))
        self.all_log_returns = np.log(self.all_closes / self.all_closes.shift(1))

    @staticmethod
    def _load_fundamental(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, engine='openpyxl')
        return (
            df.set_index(pd.to_datetime(df.iloc[:, 0]))
            .iloc[:, 1:]
            .rename_axis('date')
            .rename(columns=lambda x: x.split()[0])
        )

    @staticmethod
    def _load_closes(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, engine='openpyxl')
        df["Date"] = df["Date"].apply(lambda x: pd.Timestamp(x) if not isinstance(x, (int, float)) else pd.Timestamp('1899-12-30') + pd.Timedelta(days=x))
        return (
            df.set_index("Date")
            .rename(columns=lambda x: x.split()[0])
            .apply(pd.to_numeric, errors="coerce")
        )

    @staticmethod
    def _load_ticker_list(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, engine='openpyxl')
        return df.map(lambda x: x.split()[0] if isinstance(x, str) else x)

    def close(self, ticker: str) -> pd.Series:
        return self.all_closes[ticker]

    def log_return(self, ticker: str) -> pd.Series:
        return self.all_log_returns[ticker]