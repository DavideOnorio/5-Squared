from src.signals.momentum import Momentum
from src.data_handler.preprocessing import DataTransformer
from src.data_handler.data_handler import DataHandler
from src.signals.ranker import Ranker
from src.optimization.get_weights import Get_Weights


mo = Momentum()
a = DataTransformer()
b = DataHandler()
r = Ranker()
w = Get_Weights()

print(w.weights)