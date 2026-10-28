# Backtesting Engine + Walk-Forward ML Signal

An event-driven backtester in Python, plus an ML model whose trading signal is
evaluated **strictly out-of-sample** with walk-forward validation. It compares
Buy & Hold, an SMA crossover, and the ML strategy on the same period, after costs.

## Setup (VS Code terminal)

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Mac/Linux:           source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 8 tests should pass
```

## Run

```bash
# Real data (needs internet, downloads via yfinance and caches in data/)
python run.py --ticker SPY --start 2010-01-01 --end 2024-12-31

# Try the gradient boosting model
python run.py --ticker SPY --model gbm

# Offline smoke test (synthetic data, NEVER quote these results)
python run.py --source synthetic
```

Outputs go to `results/`: `metrics.csv`, `fold_stats.csv`, `equity_curves.csv`, `equity_curves.png`.

Useful flags: `--horizon 5`, `--threshold 0.5`, `--min-train 756`, `--test-window 63`,
`--rolling-window 756`, `--commission-bps 1`, `--slippage-bps 2`.

## How it works

```
data.py        load OHLCV (yfinance / csv / synthetic)
features.py    past-only features (returns, vol, SMA ratio, RSI, volume z-score) + forward label
walkforward.py train on the past, predict the next quarter, repeat (with label purging)
strategies.py  BuyAndHold, SMACrossover, MLSignal (long when P(up) > threshold)
engine.py      bar-by-bar loop: fill at next open, costs, mark to market at close
metrics.py     CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover, exposure
run.py         CLI that ties it together and saves results
tests/         no-look-ahead, no-leakage, cost and drawdown tests
```

### Rules that keep it honest
1. **No look-ahead in execution:** a signal made at today's close is filled at tomorrow's open.
2. **No look-ahead in features:** features at day t use only data up to day t (tested).
3. **Walk-forward, not one split:** the model is retrained every `test-window` days on past data only.
4. **Label purging:** the last `horizon` training rows are dropped, because their labels peek into the test period (tested).
5. **Costs included:** commission and slippage in bps, charged on every trade.
6. **Fair comparison:** all strategies run on the same out-of-sample period.

## Interview talking points
- Why is a single train/test split bad for time series? What is walk-forward validation?
- What is look-ahead bias? Where does this project prevent it?
- Why purge the last `horizon` training rows? (label leakage)
- Sharpe vs Sortino vs max drawdown vs Calmar; what turnover does to net returns.
- AUC ~0.5 means no edge. Daily markets are close to noise, so beating Buy & Hold after costs is hard. Reporting that honestly is a strength.
- Limits: long-only, single asset, fixed cost model, no position sizing.

## Resume bullet (fill in ONLY after running on real data)
> Built an event-driven backtesting engine in Python (next-open fills, commission + slippage); evaluated an ML signal
> with purged walk-forward validation on `<TICKER>` (`<years>` out-of-sample), reporting Sharpe `<x>`,
> max drawdown `<y>%` and turnover vs Buy & Hold and SMA baselines.

Use your real numbers. Even if the ML strategy loses to Buy & Hold, you can still say so honestly.

## Ideas to extend
- Multiple assets and a portfolio (cross-sectional ranking)
- Long/short and volatility-based position sizing
- Hyperparameter search inside the training window only (nested walk-forward)
- More features (macro, cross-asset), XGBoost/LightGBM, or an LSTM
- Parameter sensitivity and bootstrap confidence intervals on Sharpe
