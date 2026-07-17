
import streamlit as st
import requests
import os
st.header("Here you can predict the volatility for the next day using either of the four models: XGBoost (best peforming), LSTM, RNN, and Garch")

st.subheader("Ticker: Ticker (or a ticker symbol) is a short, unique code used to identify a publicly traded company or asset in a stock exchange, currently we have 20 tickers, which you can predict volatility of.")

st.markdown("**Supported Tickers:** `AAPL` • `MSFT` • `NVDA` • `GOOGL` • `AMD` • `META` • `TSLA` • `NFLX` • `LLY` • `AVGO` • `MU` • `QCOM` • `UNH` • `WMT` • `MA` • `JNJ` • `PG` • `HD` • `ORCL` • `JPM`")

st.divider()

st.subheader("Select a ticker")
ticker = st.selectbox(
    "",
    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD", "META", "TSLA", "NFLX", "LLY", "AVGO", "MU", "QCOM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "JPM"]
)
st.subheader("Select the model")
model = st.selectbox("", ['XGBoost', 'Garch', 'LSTM', 'RNN'])
st.divider()

st.write("Click the button below to predict volatility for the next day!")
API_URL = os.getenv("API_URL", "https://clarity-yqh7.onrender.com")
payload = {
    "ticker": ticker.upper(),
    "models": model.lower()
}
if st.button("Predict"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Complete")

            st.write(f"Ticker: {result['ticker']}")
            st.write(f"Model: {result['model'].upper()}")
            st.metric(
                "Predicted Volatility",
                f"{result['predicted_volatility']:.4f}"
            )

        else:
            st.error(f"{response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")      