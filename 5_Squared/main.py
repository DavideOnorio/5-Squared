from src.signals.momentum import Momentum
from src.data_handler.preprocessing import DataTransformer
from src.data_handler.data_handler import DataHandler
from src.signals.ranker import Ranker
from src.optimization.get_weights import Get_Weights
from src.backtest.portfolio import Backtest

import pandas as pd


c = DataHandler()
mo = Momentum()
r = Ranker()
w = Get_Weights()
b = Backtest()

b.run()
print(w.weights)
print(w.beta_penalized_weights, w.beta_penalized_summary)
print(w.alpha_opt_weights,w.alpha_opt_summary)
print(w.alpha_from_hrp_weights,w.alpha_from_hrp_summary)  
