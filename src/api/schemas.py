### What data are allowed to enter and leave the api !
import sys, os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
from pydantic import BaseModel 
from src.data_pipeline.ingest import watchlist

class PredictionRequest(BaseModel):
    pass
print(watchlist)