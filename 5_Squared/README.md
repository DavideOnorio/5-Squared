# 5-Squared

A quantitative equity-portfolio construction project for stock ranking, constrained optimization, backtesting, and portfolio evaluation.
The framework is built to support a full research workflow: ingesting market and fundamental data, transforming it into investable signals, constructing portfolios under maximization problem optimization, and evaluating results against the benchmark (S&P 500).

## Project Structure
```text
5-SQUARED/
└── 5_Squared/
    ├── data/
    │   └── raw/
    │       ├── beta quarterly 14y.xlsx
    │       ├── full_stocks_14y.xlsx
    │       ├── ind_5y.xlsx
    │       ├── read.py
    │       └── sep500_14y.xlsx
    │
    ├── src/
    │   ├── backtest/
    │   │   └── portfolio.py
    │   ├── data_handler/
    │   │   ├── __init__.py
    │   │   ├── data_handler.py
    │   │   └── preprocessing.py
    │   ├── optimization/
    │   │   ├── get_weights.py
    │   │   └── portfolio_metrics.py
    │   ├── signals/
    │   │   ├── __init__.py
    │   │   ├── momentum.py
    │   │   └── ranker.py
    │   ├── visual/
    |   │   ├── graphics.py
    │   │   └── metrics.py
    │   ├── __init__.py
    │   └── main.py
    ├── README.md
    └── requirements.txt
```

### Main Components
- src/data_handler/: data ingestion and preprocessing
- src/signals/: momentum and ranking logic
- src/optimization/: portfolio construction and metrics
- src/backtest/: strategy simulation
- src/visual/: charts and reporting utilities
- src/main.py: the project entry point

## Overview
The project follows a modular workflow:
- Load and preprocess market and fundamental data
- Generate momentum and factor-based signals
- Rank securities
- Optimize portfolio weights
- Run a backtest
- Evaluate performance and produce visual outputs

### Key Features
* Fundamental and momentum-driven security selection
* Transformation of data before signals are fed into the ranking stage
* Cross-sectional ranking framework for stock scoring
* Portfolio optimization with practical constraints
* Support for Sharpe-oriented and beta-aware objectives
* Historical backtesting versus a benchmark index
* Performance analytics with return and risk diagnostics
* Graphical outputs for cumulative returns, drawdowns, and portfolio behavior
* Modular codebase designed for extension and experimentation


## Data

Raw input files should be stored in:
```bash
data/raw/
```
These files include market, benchmark, and factor-related inputs used throughout the pipeline.
- individual historical asset close prices
- benchmark close prices
- risk-free rate input


## Installation
1. Clone the repository
```bash
git clone https://github.com/DavideOnorio/5-Squared
cd 5-SQUARED/5_Squared
```
2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Run
From the repository root, run:
```bash
python -m src.main
```

### Usage
A simple main.py example:
```bash
from src.backtest.portfolio import Backtest
from src.visual.metrics import PortfolioMetrics
from src.visual.graphics import PortfolioAnalytics

bt = Backtest()
bt.run()

pm = PortfolioMetrics(bt)
gr = PortfolioAnalytics(bt)

print(pm.summary())
```

## Design Principles and Good Practices
1. Keep modules single-purpose
Each model has a clearly defined responsibility as mentioned in Main Components (Project Structure)
2. Validate dimensions before optimization
Optimization bugs often come from index or shape mismatches. Before any matrix operation, verify that:
- the selected tickers match the return matrix columns
- the covariance matrix has the same ordering as the weight vector
- benchmark and asset returns share the same time index
3. Centralize assumptions
Assumptions such as lookback window, top companies to consider for the portfolio, maximum weight in the portfolio, penalty on beta, whether returns are annualized, should all be explicited in GetWeights().

## Outputs
Depending on the modules enabled, the project can generate:
- cumulative return chart of strategy versus benchmark
- drawdown chart
- portfolio summary statistics
- risk and return diagnostics
- ranking tables
- weight allocations
- comparative analytics between portfolio and benchmark

## Disclaimer
This repository is intended for educational, academic and research purposes only, as result of he 5-squared challenge #2. It should not be interpreted as investment advice, and it is not a production-ready trading system without further operational controls.

## Author
Cristina Lin
Davide Onorio
John Russel Stavale Warren
Raul Vasconcelos da Silva
Vera Lopes Nunes

LIS - Lisbon Investment Society
ISEG - Lisbon School of Economics & Management