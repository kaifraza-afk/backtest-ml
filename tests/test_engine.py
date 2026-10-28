import numpy as np
import pandas as pd

from backtest_ml.data import make_synthetic
from backtest_ml.engine import Backtester
from backtest_ml.features import build_features
from backtest_ml.metrics import max_drawdown
from backtest_ml.strategies import BuyAndHold, Strategy
from backtest_ml.walkforward import make_folds, walk_forward_predict


def _flat_df(n=10):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1e6}, index=idx)


def test_order_fills_at_next_open_not_same_close():
    df = _flat_df(5)
    df["open"] = [100, 110, 120, 130, 140]
    df["close"] = [105, 115, 125, 135, 145]
    res = Backtester(df, BuyAndHold(), 10_000, 0, 0).run()
    assert res.fills[0].date == df.index[1]
    assert res.fills[0].price == 110  # day-1 open, not day-0 close (105)


def test_costs_reduce_final_equity():
    df = make_synthetic(300)
    free = Backtester(df, BuyAndHold(), 100_000, 0, 0).run().equity.iloc[-1]
    paid = Backtester(df, BuyAndHold(), 100_000, 10, 10).run().equity.iloc[-1]
    assert paid < free


def test_buy_and_hold_trades_once():
    res = Backtester(make_synthetic(300), BuyAndHold(), 100_000, 1, 2).run()
    assert len(res.fills) == 1


def test_never_negative_equity_and_exposure_capped():
    class Flip(Strategy):
        name = "flip"

        def on_bar(self, h):
            return float(len(h) % 2)

    res = Backtester(make_synthetic(200), Flip(), 10_000, 5, 5).run()
    assert (res.equity > 0).all()
    assert (res.exposure <= 1.0 + 1e-9).all()


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 110])
    assert np.isclose(max_drawdown(eq), 90 / 120 - 1)


def test_walk_forward_has_no_label_leakage():
    horizon = 5
    for f in make_folds(2000, 756, 63, horizon):
        assert (f.train_end - 1) + horizon <= f.test_start


def test_features_do_not_use_future_data():
    df = make_synthetic(400)
    full = build_features(df)
    cut = build_features(df.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut, check_exact=False, atol=1e-10)


def test_predictions_only_in_out_of_sample_period():
    df = make_synthetic(1200)
    probs, _ = walk_forward_predict(df, "logreg", horizon=5, min_train=600, test_window=63)
    assert probs.iloc[:600].isna().all()
    assert probs.iloc[600:].notna().any()
