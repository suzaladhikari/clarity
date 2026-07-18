import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.title("Analysis of the models")
st.header("Evaluation One: MAE(Mean Absolute Error), RMSE(Root Mean Squared Error)")
model_performance_data = pd.read_parquet("./datas/modelpeformance.parquet")

### Comparison of model's performance
st.write(model_performance_data.head(5))
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
**{rmse['lstm']:.4f}**. GARCH has the highest error by a significant margin, coming in at
**{rmse['garch']:.4f}**.
"""
)