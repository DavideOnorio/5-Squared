TO WRITE

- Project title and one-liner
- Overview of the strategy
- Project structure (file tree with one-line descriptions)
- Data requirements (files, location, format, quirks)
- Installation (Python version, dependencies)
- Usage (minimal `main.py` example, configurable parameters)
- Methodology (momentum factor, Lp composite ranking, HRP allocation, score tilt, weight capping)
- Pipeline flow (DataHandler → Momentum → Ranker → GetWeights → Backtest)
- Configuration (table of key parameters with defaults)
- Results (sample output / chart description)
- Known limitations and future work

# S&P500 Alpha: Dynamic Allocation Challenge
A systematic equity strategy combining momentum signals, composite ranking, and hierarchical risk parity for robust portfolio construction.

# 1. Overview of the strategy
The strategy was developed using cross-sectional equity strategy focused on factor-based stock selection, considering core alpha driver - price momentum.
Portfolio construction integrates:
- momentum signals
- Composite ranking
- Diversification-aware allocation via HRP
- Comparison model of Sharpe ratio with penalty on beta (avoiding solely market-driven returns)
Designed to balance:
- Return (alpha generation)
- Risk (drawdown & concentration control)

# 2. Project structure

5-Squared/
│
├── data/
│   └── raw/
│        # uploaded files of data
│
├── src/
│   ├── backtest/
│   │   ├── portfolio.py               # Backtest logic & performance evaluation
│   │   └── __pycache__/
│   │
│   ├── data_handler/
│   │   ├── __init__.py
│   │   ├── data_handler.py            # Data ingestion, alignment, cleanining
│   │   ├── preprocessing.py           # Data transformations and scaling
│   │   └── __pycache__/
│   │
│   ├── optimization/
│   │   ├── get_weights.py             # Portfolio construction logic (HRP, optimization)
│   │   ├── portfolio_metrics.py       # Portfolio auxiliar metrics
│   │   ├── reasoning.md               # Methodology notes on HRP
│   │   └── __pycache__/
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── momentum.py                # Momentum signal computation
│   │   ├── ranker.py                  # Cross-sectional ranking (Lp aggregation)
│   │   └── __pycache__/
│   │
│   ├── visual/
│   │   └── __init__.py                # Visualization utilities (plots, charts)
│   │
│   └── __pycache__/
│
├── main.py                            # Pipeline entry point
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies

# 3. Data requirements (files, location, format, quirks)
Input data: weekly prices
Optional: volumes, fundamentals
Format: pandas DataFrame, Index: datetime, Columns: tickers
Location: /data/raw/ for raw xlsx files
Key quirks: survivorship bias must be addressed
Missing data: handled via filtering
Short selling: not allowed

# 4. Installation (Python version, dependencies)
Python version: Python 3.9+
Dependencies:

# 5. Methodology
1. Momentum factor
Skip most recent period to reduce reversal
Cross-sectional normalization

2. Ranking


3. HRP allocation (Hierarchical Risk Parity)
- Avoids covariance matrix inversion issues
Uses:
    1. Hierarchical clustering
    2. Quasi-diagonalization
    3. Recursive bisection
Leads to: stable allocations, better diversification.

4. Maximization of Sharpe Ratio with penalty on Beta
- Optimization rewarding high excess return per unit of total risk, but also punishing exposure to market beta.
- Maximizing Sharpe - beta_penalty*Beta <=> Minimizing -(Sharpe -  beta_penalty*Beta)
- The model computes rolling beta and rf rate (values at the formation date of the portfolio)
- Without the beta penalty, the optimizer might choose a portfolio with: strong Sharpe, but also high beta (would affect portfolio's alpha). That can happen if the assets with best expected Sharpe are also the ones most sensitive to the benchmark.
- beta_penalty controls the trade-off:
    if beta_penalty = 0, pure Sharpe maximization
    if beta_penalty small, the problem values mostly Sharpe, mild beta control
    if beta_penalty large, there is strong preference for low-beta portfolios
- parameter set at 0.05 so that the portfolio gives up a bit of Sharpe in order to reduce beta.

5. Models maximizing alpha directly
    1. pure alpha maximization
    - maximizing portfolio's alpha under weight capping
    - adding a penalty on portfolio's variance by the parameter risk_penalty
    2. alpha maximization under HRP
    - applies HRP (takes the stocks added to the portfolio based on it)
    - maximizes alpha by rebalancing the weights

# 6. Pipeline flow
    1. DataHandler (preparation of data)
    2. Momentum (feature engineering)
    3. Ranker (cross-sectional scoring)
    4. GetWeights (HRP or Sharpe)
    5. Backtest (performance evaluation)

# 7. Configurable parameters
beta_penalty: indicates how much the beta is controlled
risk_penalty: indicates how much the variance is influencing the weights optimization in the alpha optimization problem
hrp_penalty: indicates how far could be the new optimal weights to the initial HRP weights

# 8. Known limitations
    1. HRP model:
    - ignores expected returns (pure risk-based) by allocating capital based only on volatility and correlation structure
    - High-risk assets with strong expected returns may be underweighted
    - Low-return but low-vol assets may dominate
    2. Sharpe - beta model: 
    - sensitive to the beta_penalty parameter
    - does maximize sharpe ratio and control beta, but is not mathematically identical to alpha maximization
    3. Alpha models