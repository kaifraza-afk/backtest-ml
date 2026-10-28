"""Strategies. on_bar() sees history up to and including today's close and
returns the desired portfolio weight (0.0 = flat, 1.0 = fully long)."""
import pandas as pd


class Strategy:
    name = "base"

    def on_bar(self, history: pd.DataFrame) -> float:
        raise NotImplementedError


class BuyAndHold(Strategy):
    name = "Buy & Hold"

    def on_bar(self, history):
        return 1.0


class SMACrossover(Strategy):
    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow
        self.name = f"SMA {fast}/{slow}"

    def on_bar(self, history):
        if len(history) < self.slow:
            return 0.0
        close = history["close"]
        return 1.0 if close.iloc[-self.fast:].mean() > close.iloc[-self.slow:].mean() else 0.0


class MLSignal(Strategy):
    """Goes long when the walk-forward model's P(up) exceeds a threshold.
    Probabilities are precomputed out-of-sample, so no future info leaks in."""

    def __init__(self, probs: pd.Series, threshold=0.5, label="ML Signal"):
        self.probs, self.threshold, self.name = probs, threshold, label

    def on_bar(self, history):
        p = self.probs.get(history.index[-1], float("nan"))
        return 1.0 if p == p and p > self.threshold else 0.0
