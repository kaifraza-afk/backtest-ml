"""Walk-forward (out-of-sample) prediction.

For each fold we train only on the past and predict the next `test_window` days.
We also PURGE the last `horizon` training rows: their labels look `horizon` days
ahead and would otherwise peek into the test period (label leakage).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, build_labels


@dataclass
class Fold:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def make_folds(n, min_train, test_window, horizon, rolling_window=None):
    folds = []
    s = min_train
    while s < n:
        e = min(s + test_window, n)
        train_end = s - horizon + 1  # last train row t satisfies t + horizon <= s
        train_start = 0 if rolling_window is None else max(0, train_end - rolling_window)
        folds.append(Fold(train_start, train_end, s, e))
        s = e
    return folds


def make_model(name: str, seed: int = 42):
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000))])
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")


def walk_forward_predict(df, model_name="logreg", horizon=5, min_train=756,
                         test_window=63, rolling_window=None, min_samples=200, seed=42):
    """Returns (probs, fold_stats). probs[t] = P(up over next `horizon` days), known at close of t."""
    X = build_features(df)
    y = build_labels(df, horizon)
    probs = pd.Series(np.nan, index=df.index, name="prob_up")
    stats = []

    for k, f in enumerate(make_folds(len(df), min_train, test_window, horizon, rolling_window)):
        Xtr, ytr = X.iloc[f.train_start:f.train_end], y.iloc[f.train_start:f.train_end]
        ok = Xtr.notna().all(axis=1) & ytr.notna()
        Xtr, ytr = Xtr[ok], ytr[ok]
        if len(ytr) < min_samples or ytr.nunique() < 2:
            continue

        model = make_model(model_name, seed).fit(Xtr.values, ytr.values)

        Xte = X.iloc[f.test_start:f.test_end]
        ok_te = Xte.notna().all(axis=1)
        if not ok_te.any():
            continue
        p = model.predict_proba(Xte[ok_te].values)[:, 1]
        probs.loc[Xte[ok_te].index] = p

        # Fold diagnostics only where the true label is already known
        yte = y.iloc[f.test_start:f.test_end][ok_te]
        known = yte.notna()
        if known.sum() > 0:
            yk, pk = yte[known].values, p[known.values]
            auc = roc_auc_score(yk, pk) if len(np.unique(yk)) == 2 else np.nan
            stats.append({
                "fold": k,
                "train_rows": len(ytr),
                "test_start": df.index[f.test_start].date(),
                "test_end": df.index[f.test_end - 1].date(),
                "accuracy": accuracy_score(yk, pk > 0.5),
                "auc": auc,
            })
    return probs, pd.DataFrame(stats)
