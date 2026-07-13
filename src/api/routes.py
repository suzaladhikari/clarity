import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import APIRouter
from src.api.schemas import PredictionRequest, PredictedResponse
from src.api.predictivemodels import predict_volatility
router = APIRouter()

@router.post("/predict", response_model = PredictedResponse)
def predict(request: PredictionRequest):
    result = predict_volatility(request.ticker, request.models)
    return result 