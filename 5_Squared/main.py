from src.backtest.portfolio import Backtest
from src.visual.metrics import PortfolioMetrics
from src.visual.graphics import PortfolioAnalytics
from src.data_handler.data_handler import DataHandler
from src.optimization.get_weights import GetWeights

d = DataHandler()
print(d.beta)

bt = Backtest()
bt.run()

pm = PortfolioMetrics(bt)
gr = PortfolioAnalytics(bt)

print(pm.summary())

