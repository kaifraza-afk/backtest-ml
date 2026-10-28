import pandas as pd
import streamlit as st

from backtest_ml.data import load_data
from backtest_ml.engine import Backtester
from backtest_ml.metrics import metrics_table
from backtest_ml.strategies import BuyAndHold, MLSignal, SMACrossover
from backtest_ml.walkforward import walk_forward_predict

st.set_page_config(page_title="Backtesting + ML Signal", layout="wide")
st.title("Backtesting Engine + Walk-Forward ML Signal")
st.caption("Out-of-sample, after costs. Fills at next open. Not investment advice.")

with st.sidebar:
    source = st.selectbox("Data source", ["yfinance", "synthetic"])
    ticker = st.text_input("Ticker", "SPY")
    start = st.text_input("Start", "2010-01-01")
    end = st.text_input("End", "2024-12-31")
    model = st.selectbox("Model", ["logreg", "gbm"])
    horizon = st.slider("Label horizon (days)", 1, 20, 5)
    threshold = st.slider("Long if P(up) >", 0.45, 0.65, 0.50, 0.01)
    commission = st.number_input("Commission (bps)", 0.0, 50.0, 1.0)
    slippage = st.number_input("Slippage (bps)", 0.0, 50.0, 2.0)
    run = st.button("Run backtest", type="primary")


@st.cache_data(show_spinner=False)
def get_data(source, ticker, start, end):
    return load_data(source, ticker, start, end)


@st.cache_data(show_spinner=False)
def run_all(source, ticker, start, end, model, horizon, threshold, commission, slippage):
    df = get_data(source, ticker, start, end)
    min_train = 756
    probs, folds = walk_forward_predict(df, model, horizon, min_train, 63)
    strategies = [BuyAndHold(), SMACrossover(20, 50),
                  MLSignal(probs, threshold, label=f"ML ({model})")]
    results = [Backtester(df, s, 100_000, commission, slippage, start_idx=min_train).run()
               for s in strategies]
    equity = pd.concat([r.equity for r in results], axis=1)
    return metrics_table(results), equity, folds


if run:
    try:
        with st.spinner("Training walk-forward models and backtesting..."):
            table, equity, folds = run_all(source, ticker, start, end, model,
                                           horizon, threshold, commission, slippage)
    except Exception as e:
        st.error(f"Could not run: {e}")
        st.stop()

    if len(folds):
        c1, c2 = st.columns(2)
        c1.metric("Mean fold AUC", f"{folds['auc'].mean():.3f}")
        c2.metric("Mean fold accuracy", f"{folds['accuracy'].mean():.3f}")
        st.caption("AUC near 0.5 means the model has little real predictive power.")

    st.subheader("Equity curves (out-of-sample)")
    st.line_chart(equity)
    st.subheader("Metrics")
    st.dataframe(table, width="stretch")
    with st.expander("Per-fold diagnostics"):
        st.dataframe(folds, width="stretch")
else:
    st.info("Pick settings in the sidebar and click **Run backtest**.")