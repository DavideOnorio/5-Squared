from src.backtest.portfolio import Backtest
from src.visual.metrics import PortfolioMetrics
from src.visual.graphics import PortfolioAnalytics

bt = Backtest()
bt.run()

pm = PortfolioMetrics(bt)
gr = PortfolioAnalytics(bt)

print(pm.summary())

