import matplotlib.pyplot as plt

class PortfolioAnalytics:
    def __init__(self, backtest):
        self.port_return = backtest.port_return
        self.cum_portfolio = backtest.cum_portfolio
        self.cum_index = backtest.cum_index
        self.index_return = backtest.index.loc[backtest.start_date:]

    def plot_cumulative(self):
        plt.figure(figsize=(17, 8))
        plt.plot(self.cum_portfolio.index, self.cum_portfolio, label="Portfolio", linewidth=2)
        plt.plot(self.cum_index.index, self.cum_index, label="S&P 500", linewidth=2)
        plt.ylabel("Cumulative Return")
        plt.title("Portfolio vs S&P 500")
        plt.legend()
        plt.show()

    def plot_drawdown(self):
        wealth_port = 1 + self.cum_portfolio
        wealth_index = 1 + self.cum_index

        dd_port = (wealth_port - wealth_port.cummax()) / wealth_port.cummax()
        dd_index = (wealth_index - wealth_index.cummax()) / wealth_index.cummax()

        plt.figure(figsize=(17, 8))
        plt.fill_between(dd_port.index, dd_port, label="Portfolio")
        plt.fill_between(dd_index.index, dd_index, label="S&P 500")
        plt.ylabel("Drawdown")
        plt.title("Drawdown | Portfolio vs S&P 500")
        plt.legend()
        plt.show()

    def plot_all(self):
        self.plot_cumulative()
        self.plot_drawdown()