import pandas as pd
import numpy as np

df = pd.read_excel(r"5_Squared\data\raw\ind_5y.xlsx", engine='openpyxl')

# Preview
print(df)