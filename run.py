"""CLI: python run.py --ticker SPY --start 2010-01-01 --end 2024-12-31"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest_ml.data import load_data
from backtest_ml.engine import Backtester
from backtest_ml.metrics import metrics_table
from backtest_ml.strategies import BuyAndHold, MLSignal, SMACrossover
from backtest_ml.walkforward import walk_forward_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="yfinance")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--csv-path")
    ap.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    ap.add_argument("--horizon", type=int, default=5, help="label horizon in days")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-train", type=int, default=756, help="initial training rows (~3y)")
    ap.add_argument("--test-window", type=int, default=63, help="retrain every N days (~1 quarter)")
    ap.add_argument("--rolling-window", type=int, default=None, help="use rolling instead of expanding train")
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--cash", type=float, default=100_000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = load_data(a.source, a.ticker, a.start, a.end, a.csv_path)
    print(f"Loaded {len(df)} bars: {df.index[0].date()} -> {df.index[-1].date()}")

    probs, folds = walk_forward_predict(
        df, a.model, a.horizon, a.min_train, a.test_window, a.rolling_window
    )
    oos_start = a.min_train  # every strategy is compared on the SAME out-of-sample period
    print(f"Out-of-sample period starts {df.index[oos_start].date()} ({len(folds)} folds)")
    if len(folds):
        print(f"Mean fold accuracy {folds['accuracy'].mean():.3f} | mean AUC {folds['auc'].mean():.3f}"
              "  (AUC ~0.5 means no real predictive power)")

    strategies = [
        BuyAndHold(),
        SMACrossover(20, 50),
        MLSignal(probs, a.threshold, label=f"ML ({a.model})"),
    ]
    results = [
        Backtester(df, s, a.cash, a.commission_bps, a.slippage_bps, start_idx=oos_start).run()
        for s in strategies
    ]

    table = metrics_table(results)
    print("\n" + table.to_string())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv")
    folds.to_csv(out / "fold_stats.csv", index=False)
    pd.concat([r.equity for r in results], axis=1).to_csv(out / "equity_curves.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        ax1.plot(r.equity.index, r.equity.values, label=r.name)
        ax2.plot(r.equity.index, (r.equity / r.equity.cummax() - 1).values * 100, alpha=0.8)
    ax1.set_title(f"{a.ticker}: out-of-sample equity (after costs)")
    ax1.set_ylabel("Portfolio value")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png", dpi=130)
    print(f"\nSaved metrics.csv, fold_stats.csv, equity_curves.csv, equity_curves.png in {out}/")


if __name__ == "__main__":
    main()
