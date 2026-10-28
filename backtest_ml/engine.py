"""Event-driven backtester.

Timeline for each bar i:
  1. Fill any order created yesterday at TODAY'S OPEN (plus slippage + commission).
  2. Mark the portfolio to market at today's close.
  3. Ask the strategy for a target weight using data up to today's close.
     That decision is executed at tomorrow's open, never at today's close (no look-ahead).
"""
import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Fill:
    date: pd.Timestamp
    shares: int  # +buy / -sell
    price: float  # execution price after slippage
    cost: float  # commission paid


class Portfolio:
    def __init__(self, cash, commission_bps, slippage_bps):
        self.cash = float(cash)
        self.shares = 0
        self.c = commission_bps / 1e4
        self.s = slippage_bps / 1e4
        self.fills: list[Fill] = []

    def equity(self, price):
        return self.cash + self.shares * price

    def rebalance(self, date, open_price, target_weight):
        """Long-only, whole shares. Moves toward target_weight * equity."""
        target_shares = target_weight * self.equity(open_price) / open_price
        if target_shares >= self.shares:
            delta = math.floor(target_shares - self.shares)
        else:
            delta = -math.floor(self.shares - target_shares)
        if delta > 0:
            price = open_price * (1 + self.s)
            affordable = math.floor(self.cash / (price * (1 + self.c)))
            delta = min(delta, affordable)
        elif delta < 0:
            price = open_price * (1 - self.s)
            delta = -min(-delta, self.shares)
        else:
            return
        if delta == 0:
            return
        notional = abs(delta) * price
        cost = notional * self.c
        self.cash -= delta * price + cost
        self.shares += delta
        self.fills.append(Fill(date, delta, price, cost))


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    exposure: pd.Series
    fills: list = field(default_factory=list)


class Backtester:
    def __init__(self, data, strategy, initial_cash=100_000, commission_bps=1.0,
                 slippage_bps=2.0, start_idx=0):
        self.data, self.strategy = data, strategy
        self.initial_cash = initial_cash
        self.commission_bps, self.slippage_bps = commission_bps, slippage_bps
        self.start_idx = start_idx

    def run(self) -> BacktestResult:
        pf = Portfolio(self.initial_cash, self.commission_bps, self.slippage_bps)
        n = len(self.data)
        opens, closes = self.data["open"].values, self.data["close"].values
        dates = self.data.index
        equity, exposure = [], []
        pending, last_target = None, None

        for i in range(self.start_idx, n):
            if pending is not None:
                pf.rebalance(dates[i], opens[i], pending)
                pending = None

            eq = pf.equity(closes[i])
            equity.append(eq)
            exposure.append(pf.shares * closes[i] / eq if eq > 0 else 0.0)

            target = float(self.strategy.on_bar(self.data.iloc[: i + 1]))
            if target != last_target and i < n - 1:
                pending, last_target = target, target

        idx = dates[self.start_idx:]
        return BacktestResult(
            name=self.strategy.name,
            equity=pd.Series(equity, index=idx, name=self.strategy.name),
            exposure=pd.Series(exposure, index=idx),
            fills=pf.fills,
        )
