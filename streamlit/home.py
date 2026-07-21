import streamlit as st
st.title("Clarity: A weather forecast for market turbulence")

st.header("What is Clarity ?")
# st.set_page_config(
#     page_title="Clarity",
#     page_icon="📈",
#     layout="wide"
# )
st.subheader("Clarity does not try to predict where the market goes tomorow. What it predicts is how much is the stock likely to swing, in either direction. That's volatility and unlike the price detection it's actually forecastable.")

### FAQ 2 
st.divider()

st.header("Why does it matter?")

st.subheader("Volatilty is not a side stat - its the metric a lot of financial companies rely on. Option traders uses volatilty mostly to trade off the price contracts. Better volatility decisions leads to better decisions with money.")

st.divider()
st.header("What's under the hood?")
st.subheader("Clarity benchmarks four different approaches on real data across 20 large-cap stocks plus the S&P 500")
st.write("GARCH: The traditional statistical model that assumes that the volatility remebers yesterday and reacts the same way to good news and bad news")
st.write("XGBoost: A gradient-boosted tree model")
st.write("LSTM and RNN: Deep learning models built up on longer, more complex patterns the GARCH's maths cannot see.")

st.divider()

st.header("Headline result:")
st.subheader("XGBoost: R² = 0.75 vs. GARCH: R² = 0.55")

st.divider()
st.write("To check how the model is trained and tested, please click the button below")
st.link_button(url ="https://github.com/suzaladhikari/clarity.git", label="GitHub Repo")