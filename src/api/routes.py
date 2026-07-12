import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import APIRouter
from src.api.schemas import PredictionRequest, PredictedResponse
router = APIRouter()

print(PredictedResponse)
print(PredictionRequest)
