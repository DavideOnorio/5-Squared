import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "data.csv"

import numpy as np
import pandas as pd
from pathlib import Path

# ... (PROJECT_ROOT and DEFAULT_CSV definitions stay the same)

class DataHandler:
    def __init__(self, source=None):
        if source is None:
            if not DEFAULT_CSV.exists():
                raise FileNotFoundError(f"Could not find data at {DEFAULT_CSV}.")
            source = CSVLoader(csv_path=DEFAULT_CSV)
            
        self.all_closes, self.all_volumes = source.load()
        self.all_log_returns = np.log(self.all_closes).diff()
        self.fundamental = pd.read_excel(r"5_Squared\data\raw\Book1.xlsx", engine='openpyxl')
        self.fundamental = self.fundamental.set_index('Ticker').rename(columns=lambda x: x.split()[0])

    def close(self, ticker) -> pd.Series:
        return self.all_closes[ticker]

    def volume(self, ticker) -> pd.Series:
        return self.all_volumes[ticker]

    def log_return(self, ticker) -> pd.Series:
        """Returns the log percentage change for a specific ticker"""
        return self.all_log_returns[ticker]


class CSVLoader:
    def __init__(self, csv_path=DEFAULT_CSV):
        self.csv_path = csv_path

    def load(self):
        raw = pd.read_csv(self.csv_path, header=None, index_col=0, low_memory=False)

        multi = pd.MultiIndex.from_arrays(
            [raw.iloc[0].values, raw.iloc[1].values]
        )

        data = raw.iloc[3:].copy()
        data.index = pd.to_datetime(data.index, errors="coerce")
        data.columns = multi
        data = data.apply(pd.to_numeric, errors="coerce")

        data = data.dropna(how='all')

        closes  = data.xs("Close",  axis=1, level=0)
        volumes = data.xs("Volume", axis=1, level=0)

        return closes, volumes
    
    