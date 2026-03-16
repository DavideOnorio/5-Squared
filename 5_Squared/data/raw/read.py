import pandas as pd

df = pd.read_excel(r"5_Squared\data\raw\Book1.xlsx", engine='openpyxl')

df.columns = [col.split()[0] for col in df.columns]
print(df)