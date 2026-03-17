import pandas as pd
import numpy as np

df = pd.read_excel(r"5_Squared\data\raw\sep500_5y.xlsx")


# Convert Excel serial dates to readable weekly dates
df["Date"] = pd.to_datetime(df["Date"], unit="D", origin="1899-12-30")

# Set Date as index
df = df.set_index("Date")
df.index = df.index.to_period("W").to_timestamp()

# Keep only the first word of each ticker
df.columns = df.columns.str.split().str[0]
df = df.apply(pd.to_numeric, errors="coerce")
returns = np.log(df / df.shift(1))
print(returns)