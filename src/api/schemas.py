### What data are allowed to enter and leave the api !
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pydantic import BaseModel, field_validator
from src.data_pipeline.ingest import watchlist
class PredictionRequest(BaseModel):
    ticker : str 
    @field_validator("name")
    @classmethod
    def validate_name(cls,value):
        if value not in watchlist.keys():
            raise ValueError(f"{value} is not in the list of stocks data we have. Choost from {list(watchlist.keys())} we have")
        return value 
    