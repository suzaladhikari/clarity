import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
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

### Datas
lstm_rnn_results = pd.read_json('./modelperformance/lstm_rnn_predicted.json')
xgboost_results = pd.read_json('./modelperformance/xbgoost_predicted.json')
garch_results = pd.read_json('./modelperformance/garch_predicted.json')
combined_data = pd.read_parquet('./datas/combined_data.parquet') 

st.divider()
st.header("Evaluation Three: Actual vs Predicted Volatility")

garch_results = garch_results.drop(columns='model')
xgboost_results = xgboost_results.drop(columns='model')

combined_data = combined_data[combined_data['date'] >= '2022-01-01']
# Reset indexes first
combined_data = combined_data.reset_index(drop=True)

xgboost_results = xgboost_results.reset_index(drop=True)

garch_results = garch_results.reset_index(drop=True)


# Assign values
combined_data["xgboost_predicted"] = (
    xgboost_results["xgboost_predicted"].values
)

combined_data["garch_predicted"] = (
    garch_results["garch_predicted"].values
)

lstm_results = lstm_rnn_results[lstm_rnn_results['model'] == 'lstm'].drop(columns = 'predicted_rnn')
rnn_results = lstm_rnn_results[lstm_rnn_results['model'] == 'rnn'].drop(columns = 'predicted_lstm')



lstm_results["predicted_lstm"] = lstm_results["predicted_lstm"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Make each prediction its own row
lstm_results = lstm_results.explode("predicted_lstm", ignore_index=True)

# Convert to float
lstm_results["predicted_lstm"] = lstm_results["predicted_lstm"].astype(float)
lstm_results = lstm_results.drop(columns = 'model')

rnn_results["predicted_rnn"] = rnn_results["predicted_rnn"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Make each prediction its own row
rnn_results = rnn_results.explode("predicted_rnn", ignore_index=True)

# Convert to float
rnn_results["predicted_rnn"] = rnn_results["predicted_rnn"].astype(float)
rnn_results = rnn_results.drop(columns = 'model')


combined_data['lstm_predicted'] = lstm_results['predicted_lstm']
combined_data['rnn_predicted'] = rnn_results['predicted_rnn']

plot_data = combined_data.dropna(subset=["target_volatility","xgboost_predicted","garch_predicted","lstm_predicted","rnn_predicted"]).reset_index(drop=True)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=plot_data, x="date", y="target_volatility", ax=ax[0,0], s=10, color="blue", label="Actual")
sns.scatterplot(data=plot_data, x="date", y="xgboost_predicted", ax=ax[0,0], s=10, color="red", label="XGBoost")
ax[0,0].set_title("XGBoost vs Actual")


sns.scatterplot(data=plot_data, x="date", y="target_volatility", ax=ax[0,1], s=10, color="blue", label="Actual")
sns.scatterplot(data=plot_data, x="date", y="garch_predicted", ax=ax[0,1], s=10, color="green", label="GARCH")
ax[0,1].set_title("GARCH vs Actual")


sns.scatterplot(data=plot_data, x="date", y="target_volatility", ax=ax[1,0], s=10, color="blue", label="Actual")
sns.scatterplot(data=plot_data, x="date", y="lstm_predicted", ax=ax[1,0], s=10, color="pink", label="LSTM")
ax[1,0].set_title("LSTM vs Actual")


sns.scatterplot(data=plot_data, x="date", y="target_volatility", ax=ax[1,1], s=10, color="blue", label="Actual")
sns.scatterplot(data=plot_data, x="date", y="rnn_predicted", ax=ax[1,1], s=10, color="yellow", label="RNN")
ax[1,1].set_title("RNN vs Actual")


for a in ax.flat:
    a.set_xlabel("Date")
    a.set_ylabel("Volatility")
    a.tick_params(axis="x", rotation=45)
    a.legend()
    a.grid(True)

plt.tight_layout()
st.pyplot(fig)