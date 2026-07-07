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
from src.training.train_xgboost import xgboost_model

def extracting_to_final_data(key):
    correctedData = data_corrector(key)
    feature_engineering = feature_creation(correctedData)
    return feature_engineering

combined_data = []
for ticker in watchlist.keys():
    stock_data = extracting_to_final_data(ticker)
    combined_data.append(stock_data)

combined_stock_data = pd.concat(combined_data, ignore_index=True)

## Creating the sequences out of the combined datasets
X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(combined_stock_data)
train_loader, validation_loader, test_loader = creating_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

## Models 
lstm_model_results = lstm_model(train_loader, validation_loader, test_loader)
rnn_model_results = rnn_model(train_loader, validation_loader, test_loader)
xgboost_resutls = xgboost_model(combined_stock_data)

print(xgboost_resutls['rmse'])

## Ani tes pachi make sure you save the whole files in the dictionary ! got it 
## Ani __name__ == __main__ ko use ni k huncha bhanera bujne 
## Ani make sure you create the reusable function in seqeucen builder and also the dataloader to carry on your project