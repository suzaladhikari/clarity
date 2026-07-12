import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import FastAPI 
from src.api.routes import router
app = FastAPI(title = "API for Clarity", description= "API for stock volatility prediction", version = "1.0.0")
#uvicorn src.api.main:app --reload
# Creating the routers 
@app.get('/info')
def say_hello():
    return "Through Clarity API you can predict the volatility for 20 different stocks using four different models: LSTM, RNN, GARCH, or XGBoost"

## Registering all the routers created in routes.py 
app.include_router(router)
