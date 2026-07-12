### What data are allowed to enter and leave the api !
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pydantic import BaseModel, field_validator
from src.data_pipeline.ingest import watchlist

model_list = {
    'xgboost': 'XGBoost Regressor',
    'lstm': 'lstm model',
    'rnn': 'rnn model',
    'garch': 'garch model'

}
class PredictionRequest(BaseModel):
    ticker : str 
    @field_validator("ticker")
    @classmethod
    def validate_name(cls,value):
        if value not in watchlist.keys():
            raise ValueError(f"{value} is not in the list of stocks data we have. Choost from {list(watchlist.keys())} we have")
        return value 
    
    ### Models list 
    models: str 
    @field_validator("models")
    @classmethod
    def validate(cls, value):
        if value not in model_list.keys():
            raise ValueError(f"{value} is not in the list of models we have. Choose from {list(model_list.keys())} we have")


class PredictedResponse(BaseModel):
    ticker: str
    predicted_volatility = str
    model : str 
