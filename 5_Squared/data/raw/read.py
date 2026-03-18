import pandas as pd
import numpy as np

df = pd.read_excel(r"5_Squared\data\raw\removed_tickers_sp500 _5y.xlsx", engine='openpyxl')
df = df.map(lambda x: x.split()[0] if isinstance(x, str) else x)

print(df[2022])

