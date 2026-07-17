
import streamlit as st

st.title("📊 Dashboard")

ticker = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "TSLA"]
)

st.write(f"Prediction for {ticker}")