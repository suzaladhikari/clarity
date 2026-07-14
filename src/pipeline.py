## Creating the pipeline so that I donot have to rerun the code over again and again 
## Necessary imports 
import sys 
import os 
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_pipeline.ingest import watchlist ## Function to extract the company dataset 
from src.data_pipeline.cleaner import data_corrector ## to correct and check the nans and infs
from src.data_pipeline.featureengineering import feature_creation ## Creating new features
from src.datasets.sequence_builder import train_test_val
from src.datasets.dataset_loader import creating_data_loaders
from src.training.train_lstm import lstm_model
from src.training.train_rnn import rnn_model 
import json 
import numpy as np 
def extracting_to_final_data(key):
    correctedData = data_corrector(key)
    feature_engineering = feature_creation(correctedData,training=True)
    return feature_engineering

combined_data = []
for ticker in watchlist.keys():
    stock_data = extracting_to_final_data(ticker)
    combined_data.append(stock_data)

combined_stock_data = pd.concat(combined_data, ignore_index=True)
combined_stock_data.to_parquet('./datas/combined_data.parquet')
print(combined_stock_data['date'].max())
    ## Creating the sequences out of the combined datasets
def lstm_rnn(combined_stock_data):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(combined_stock_data)
    train_loader, validation_loader, test_loader = creating_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        ## Models 
    lstm_model_results = lstm_model(train_loader, validation_loader, test_loader)
    rnn_model_results = rnn_model(train_loader, validation_loader, test_loader)  
    combined_results = [
        {
            "model": "lstm",
            "mae": lstm_model_results["mae"],
            "rmse": lstm_model_results["rmse"],
            "r2": lstm_model_results["r2"],
        },
        {
            "model": "rnn",
            "mae": rnn_model_results["mae"],
            "rmse": rnn_model_results["rmse"],
            "r2": rnn_model_results["r2"],
        },
    ]

    return combined_results

lstm_rnn_results = lstm_rnn(combined_stock_data)

path = "./modelperformance/lstm_rnn_results.json"

os.makedirs(os.path.dirname(path), exist_ok=True)

with open(path, "w") as f:
    json.dump(lstm_rnn_results, f, indent=4)
