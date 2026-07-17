
import streamlit as st
import requests

st.header("Here you can predict the volatility for the next day using either of the four models: XGBoost (best peforming), LSTM, RNN, and Garch")

st.subheader("Ticker: Ticker (or a ticker symbol) is a short, unique code used to identify a publicly traded company or asset in a stock exchange, currently we have 20 tickers, which you can predict volatility of.")

st.markdown("**Supported Tickers:** `AAPL` • `MSFT` • `NVDA` • `GOOGL` • `AMD` • `META` • `TSLA` • `NFLX` • `LLY` • `AVGO` • `MU` • `QCOM` • `UNH` • `WMT` • `MA` • `JNJ` • `PG` • `HD` • `ORCL` • `JPM`")

st.divider()

ticker = st.selectbox(
    "Choose a ticker",
    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD", "META", "TSLA", "NFLX", "LLY", "AVGO", "MU", "QCOM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "JPM"]
)