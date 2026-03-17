from src.optimization.get_weights import Get_Weights
from src.data_handler.data_handler import DataHandler

class Backtest:
    def __init__(self):
        self.w = Get_Weights()
        self.d = DataHandler()
        self.weights = self.w.weights
        self.df = self.d.all_closes
    
    

