import pandas as pd
import numpy as np

df = pd.read_excel(r"5_Squared\data\raw\fundam_5y.xlsx", engine='openpyxl')
df = df.set_index(df.columns[0])
df.index = pd.to_datetime(df.index)
df.index.name = 'date'  # optional: rename it

# Clean ticker columns
df.columns = df.columns.str.extract(r'^(\S+)')[0]
# Preview
print(df)