from src.signals.momentum import Momentum
from src.data_handler.preprocessing import DataTransformer
from src.data_handler.data_handler import DataHandler
from src.signals.ranker import Ranker
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

mo = Momentum()
a = DataTransformer()
b = DataHandler()
r = Ranker()

print(r.score)