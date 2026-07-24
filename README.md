# 📈 Clarity: A weather forecast for market turbulence

## What is Clarity ?

Stock volatility forecasting platform combining classical statistical models with machine learning, served through a FastAPI backend and a Streamlit frontend.

---
## Project Overview 

Clarity is an end-to-end platform for stock volatility forecasting, built to compare classical and machine-learning approaches under one roof. It runs four models in parallel — XGBoost, LSTM, RNN, and GARCH — refreshed daily through an automated data pipeline. The system is fully deployed: a Dockerized FastAPI backend on Render, and a Streamlit frontend for interactive exploration of predictions and model performance.

## 🌐 Live Demo

👉 **[Try the app here](https://claritystockvolatiltypredictor.streamlit.app/)**

The application is live and hosted on Streamlit Cloud. No installation required — just open the link and start predicting!

---

## Project Structure 

```
clarity/
│
├── streamlit/                        # Streamlit Frontend
│   ├── home.py                       # Main landing page
│   └── pages/
│       ├── 1_Predictor.py            # Volatility prediction interface
│       ├── 2_Analysis_of_Models.py   # Model comparison and evaluation views
│       └── 3_Developer_Details.py    # Project/developer info page
│
├── src/                              # Core application code
│   ├── api/                          # FastAPI Backend
│   │   ├── main.py                   # API entry point
│   │   ├── routes.py                 # Prediction endpoints
│   │   ├── predictivemodels.py       # Model loading and inference logic
│   │   └── schemas.py                # Request/response schemas (Pydantic)
│   │
│   ├── data_pipeline/                # Data ingestion and preprocessing
│   │   ├── ingest.py                 # Raw data collection
│   │   ├── cleaner.py                # Data cleaning
│   │   ├── featureengineering.py     # Feature construction
│   │   └── sentimentanalysis.py      # Sentiment feature extraction
│   │
│   ├── datasets/                     # Dataset preparation utilities
│   │   ├── dataset_loader.py         # Loads processed datasets
│   │   └── sequence_builder.py       # Builds sequences for LSTM/RNN input
│   │
│   ├── models/                       # Model architectures
│   │   ├── garch.py                  # GARCH(1,1) implementation
│   │   ├── lstm.py                   # LSTM model
│   │   ├── rnn.py                    # RNN model
│   │   └── xgboost.py                # XGBoost model
│   │
│   ├── training/                     # Model training scripts
│   │   ├── train_garch.py
│   │   ├── train_lstm.py
│   │   ├── train_rnn.py
│   │   ├── train_xgboost.py
│   │   └── trainer_utils.py          # Shared training utilities
│   │
│   ├── utils/                        # Shared helpers
│   │   ├── initials.py
│   │   ├── metrics.py                # R², RMSE, MAE, etc.
│   │   └── seed.py                   # Reproducibility (random seed control)
│   │
│   ├── pipeline.py                   # End-to-end pipeline orchestration
│   └── runall.py                     # Entry point to run the full pipeline
│
├── datas/                            # Data files
│   ├── raw/                          # Raw per-ticker data (20 stocks)
│   ├── processed/                    # Cleaned, processed data
│   ├── features/                     # Feature-engineered data
│   ├── combined_data.parquet         # Combined dataset across tickers
│   └── modelpeformance.parquet       # Aggregated model performance
│
├── models_saved/                     # Trained model artifacts
│   ├── lstm/lstm_best.pt
│   ├── rnn/rnn_best.pt
│   ├── xgboost/xgboost_model.json
│   └── scaler.pkl                    # Fitted feature scaler
│
├── modelperformance/                 # Evaluation outputs
│   ├── garch_params.json
│   ├── garch_predicted.json
│   ├── lstm_rnn_predicted.json
│   ├── xbgoost_predicted.json
│   ├── performancefiles/             # Per-model results (JSON)
│   ├── gperformancefiles/            # ARCH model results
│   ├── r2scores.png                  # R² comparison chart
│   └── maeandrmsescores.png          # MAE/RMSE comparison chart
│
├── logs/
│   └── runall.log                    # Pipeline execution logs
│
├── Dockerfile.fastapi                # Dockerfile for FastAPI backend
├── Dockerfile.streamlit              # Dockerfile for Streamlit frontend
├── docker-compose.yml                # Multi-container orchestration
├── environment.yml                   # Conda environment spec
├── requirements.txt                  # Base dependencies
├── requirements-fastapi.txt          # Backend dependencies (local)
├── requirements-fastapi-docker.txt   # Backend dependencies (Docker)
└── README.md
```

---
