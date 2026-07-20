import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.title("Analysis of the models")
st.header("Evaluation One: MAE(Mean Absolute Error), RMSE(Root Mean Squared Error)")
model_performance_data = pd.read_parquet("./datas/modelpeformance.parquet")

### Comparison of model's performance
fig, ax = plt.subplots(1,2,figsize = (15,10))
def plottingmetrics(x,y, data, title, xlabel, ylabel, ylim1, ylim2, axes):
    sns.barplot(x = x, y =y, data = data, hue = x, ax = axes)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_ylim(ylim1, ylim2)
    for container in axes.containers:
        axes.bar_label(container, fmt="%.4f", padding=3)

plottingmetrics('model', 'mae', model_performance_data, 'Comparison of MAE Scores', 'Models', 'MAE Score', 0, 0.01, ax[0])
plottingmetrics('model', 'rmse', model_performance_data, 'Comparison of RMSE Scores', 'Models', 'RMSE Score', 0, 0.01, ax[1])
plt.tight_layout()
st.pyplot(fig)

st.subheader("Comparison of MAE Scores (Left)")
mae = model_performance_data.set_index("model")["mae"]

st.write(
    f"""
XGBoost achieves the best (lowest) performance with an MAE of **{mae['xgboost']:.4f}**, followed by
RNN (**{mae['rnn']:.4f}**). LSTM performs similarly (**{mae['lstm']:.4f}**), while
GARCH has the highest error (**{mae['garch']:.4f}**).
"""
)

st.subheader("Comparison of RMSE Scores (Right)")
rmse = model_performance_data.set_index("model")["rmse"]

st.write(
    f"""
XGBoost achieves the best (lowest) performance with an RMSE of **{rmse['xgboost']:.4f}**, followed by
RNN with an RMSE of **{rmse['rnn']:.4f}**. LSTM performs similarly to the RNN with an RMSE of
**{rmse['lstm']:.4f}**. GARCH undeperforms by a significant margin, coming in at
**{rmse['garch']:.4f}**.
"""
)

st.divider()
st.header("Evaluation Two: R2 Scores")

r2 = model_performance_data.set_index("model")["r2"]

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    x="model",
    y="r2",
    data=model_performance_data,
    ax=ax
)

ax.set_xlabel("Models")
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1.05)
ax.set_title("Comparison of R² Scores Across Models")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)

st.pyplot(fig)

st.write(
    f"""
**XGBoost ({r2['xgboost']:.3f})**: Once again, XGBoost takes the top spot. It explains approximately **{r2['xgboost']*100:.1f}%** of the variance in the target variable, reinforcing its position as the strongest model in this comparison.

**RNN ({r2['rnn']:.3f})**: The standard RNN secures the second position, explaining **{r2['rnn']*100:.1f}%** of the variance.

**LSTM ({r2['lstm']:.3f})**: The LSTM follows closely behind the RNN, explaining **{r2['lstm']*100:.1f}%** of the variance.

**GARCH ({r2['garch']:.3f})**: The statistical GARCH model lags significantly behind the machine learning approaches, explaining only **{r2['garch']*100:.1f}%** of the variance.
"""
)

alex = pd.read_parquet("./datas/combined_data.parquet")

simran = alex[alex['date'] >= '2022-01-01']
st.write(simran['date'].dtype)
fig, ax = plt.subplots(figsize=(12, 5))
sns.scatterplot(
    x="date",
    y="target_volatility",
    data=simran,
    ax=ax
)

st.pyplot(fig)


data = pd.read_json('./modelperformance/xgboost_predicted.json')
print(data.head(5))