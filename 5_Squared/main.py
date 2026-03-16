from src.signals.momentum import Momentum
from src.data_handler.preprocessing import DataTransformer

mo = Momentum()
a = DataTransformer()
print(a.scale(mo.momentum_factor()))