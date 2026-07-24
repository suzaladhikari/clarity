# Clarity

Stock volatility forecasting platform comparing classical statistical modeling against machine learning, served through a FastAPI backend and a Streamlit frontend.

**Live app:** [claritystockvolatiltypredictor.streamlit.app](https://claritystockvolatiltypredictor.streamlit.app/)
**API docs:** `https://clarity-yqh7.onrender.com/docs`
**Source:** [github.com/suzaladhikari/clarity](https://github.com/suzaladhikari/clarity)

---

## Project Overview

Clarity forecasts short-term stock volatility across 20 large-cap stocks, benchmarking a classical econometric model (GARCH(1,1)) against three machine learning approaches (XGBoost, LSTM, RNN). The goal is to test whether ML models can meaningfully outperform a well-established statistical baseline on volatility prediction, and to expose that comparison through a live, interactive tool rather than a static notebook.

The system is fully productionized: a daily automated pipeline ingests and processes market data, four models generate predictions, a FastAPI service exposes them over REST, and a Streamlit app renders live forecasts and model comparisons.

---

## Model Performance

| Model   | R² Score |
|---------|----------|
| XGBoost | 0.750    |
| RNN     | 0.715    |
| LSTM    | 0.707    |
| GARCH   | 0.554    |


GARCH(1,1) serves as the classical baseline; all three ML models outperform it on this dataset, with XGBoost currently leading.

**Coverage:** AAPL, MSFT, GOOGL, NVDA, TSLA, META, AMD, AVGO, MU, ORCL, QCOM, JPM, MA, HD, WMT, PG, JNJ, LLY, UNH, NFLX, plus SPY as a market benchmark.

---

## Key Features

- **Multi-model comparison** — XGBoost, LSTM, RNN, and GARCH(1,1) predictions generated side by side for the same tickers
- **Automated data pipeline** — daily ingestion, cleaning, and feature engineering, including sentiment-based features, with no manual retraining step
- **FastAPI backend** — REST endpoints for predictions and model performance, with auto-generated Swagger docs
- **Streamlit frontend** — multi-page app with a predictor view, a model analysis/comparison view, and a project details page
- **Dockerized deployment** — separate Dockerfiles for backend and frontend, orchestrated via Docker Compose for local development
- **Cloud deployment** — backend on Render, frontend on Streamlit Community Cloud

---

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)

---

## Running with Docker (Recommended)

**1. Clone the repository**

```bash
git clone https://github.com/suzaladhikari/clarity.git
cd clarity
```

**2. Build and start all containers**

```bash
docker compose up --build
```

**3. Access the application**

| Service       | URL                          |
|---------------|-------------------------------|
| Streamlit App | http://localhost:8501         |
| FastAPI Docs  | http://localhost:8000/docs    |

**4. Stop all containers**

```bash
docker compose down
```

---

## Running Locally (Without Docker)

**1. Clone the repository**

```bash
git clone https://github.com/suzaladhikari/clarity.git
cd clarity
```

**2. Install backend dependencies**

```bash
pip install -r requirements-fastapi.txt
```

**3. Install frontend dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the FastAPI backend**

```bash
uvicorn src.api.main:app --reload
```

**5. Start the Streamlit frontend** (in a new terminal)

```bash
streamlit run streamlit/home.py
```

**6. Access the application**

| Service       | URL                          |
|---------------|-------------------------------|
| Streamlit App | http://localhost:8501         |
| FastAPI Docs  | http://localhost:8000/docs    |

---

## How the System Works

```
Daily cron trigger
       |
Data ingestion (src/data_pipeline/ingest.py)
       |
Cleaning + feature engineering (cleaner.py, featureengineering.py, sentimentanalysis.py)
       |
Model inference: XGBoost | LSTM | RNN | GARCH(1,1)
       |
FastAPI backend serves predictions (src/api)
       |
Streamlit frontend queries API and renders forecasts
```

1. A scheduled job pulls fresh market data for all 20 tickers
2. Data is cleaned, engineered into features, and enriched with sentiment signals
3. All four models generate updated volatility forecasts
4. Results are served via FastAPI endpoints
5. The Streamlit app queries the API and displays predictions and model comparisons

---

## Project Structure

```
clarity/
├── streamlit/                        # Streamlit frontend
│   ├── home.py
│   └── pages/
│       ├── 1_Predictor.py
│       ├── 2_Analysis_of_Models.py
│       └── 3_Developer_Details.py
│
├── src/
│   ├── api/                          # FastAPI backend
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── predictivemodels.py
│   │   └── schemas.py
│   ├── data_pipeline/                # Ingestion, cleaning, feature engineering
│   ├── datasets/                     # Dataset + sequence loaders for LSTM/RNN
│   ├── models/                       # Model architectures
│   ├── training/                     # Training scripts per model
│   ├── utils/                        # Metrics, seeding, shared helpers
│   ├── pipeline.py
│   └── runall.py
│
├── datas/                            # Raw, processed, and feature data
├── models_saved/                     # Trained model artifacts
├── modelperformance/                 # Evaluation results and charts
├── logs/
│
├── Dockerfile.fastapi
├── Dockerfile.streamlit
├── docker-compose.yml
├── environment.yml
├── requirements.txt
├── requirements-fastapi.txt
├── requirements-fastapi-docker.txt
└── README.md
```

---

## Project Goals

- Rigorously compare classical (GARCH) and machine learning (XGBoost, LSTM, RNN) approaches to volatility forecasting on identical data
- Build a complete MLOps pipeline: automated ingestion, training, evaluation, and serving
- Containerize and deploy the full stack for public, real-time access
- Apply the same reproducibility discipline (fixed seeds, tracked metrics) used in research to a deployed product

---

---

## Contact
 
**Sujal Adhikari**
[Email](mailto:sujaladhikarids@gmail.com) · [LinkedIn](https://www.linkedin.com/in/sujaladhikari3/) · [GitHub](https://github.com/suzaladhikari)