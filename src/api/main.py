from fastapi import FastAPI 

app = FastAPI(title = "API for Clarity", description= "API for stock volatility prediction", version = "1.0.0")
#uvicorn src.api.main:app --reload
# Basic information of the api  
@app.get('/info')
def say_hello():
    return "Through Clarity API you can predict the volatility for 20 different stocks using four different models: LSTM, RNN, GARCH, or XGBoost"