import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def compute_metrics(result) -> dict:
    eq = result.equity
    rets = eq.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    std = rets.std()
    downside = rets[rets < 0].std()
    mdd = max_drawdown(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    traded = sum(abs(f.shares) * f.price for f in result.fills)
    return {
        "Strategy": result.name,
        "Total Return %": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "CAGR %": cagr * 100,
        "Volatility %": std * np.sqrt(TRADING_DAYS) * 100,
        "Sharpe": rets.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "Sortino": rets.mean() / downside * np.sqrt(TRADING_DAYS) if downside and downside > 0 else 0.0,
        "Max Drawdown %": mdd * 100,
        "Calmar": cagr / abs(mdd) if mdd < 0 else 0.0,
        "Turnover (x/yr)": traded / eq.mean() / years if years > 0 else 0.0,
        "Trades": len(result.fills),
        "Exposure %": result.exposure.mean() * 100,
    }


def metrics_table(results) -> pd.DataFrame:
    return pd.DataFrame([compute_metrics(r) for r in results]).set_index("Strategy").round(3)
