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
This project was designed to implement a systematic strategy for an equity-portfolio optimization problem and incorporates an innovative allocation method by scoring the constituents of the S&P 500 as of the start-date of the back-test, and optimizing

# 1. 