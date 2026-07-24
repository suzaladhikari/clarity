# 📈 Clarity: A weather forecast for market turbulence

## What is Clarity ?
Stock volatility forecasting platform combining classical statistical models with machine learning, served through a FastAPI backend and a Streamlit frontend.

---
### Project Overview 

Clarity is an end-to-end platform for stock volatility forecasting, built to compare classical and machine-learning approaches under one roof. It runs four models in parallel — XGBoost, LSTM, RNN, and GARCH — refreshed daily through an automated data pipeline. The system is fully deployed: a Dockerized FastAPI backend on Render, and a Streamlit frontend for interactive exploration of predictions and model performance.