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
from src.training.train_garch import garch_model
def main():
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
    garch_results = garch_model(combined_stock_data)

    combined_results = {}
    combined_results['mae'] = [lstm_model_results['mae'], rnn_model_results['mae'], xgboost_resutls['mae'], garch_results['mae']]
    combined_results['rmse'] = [lstm_model_results['rmse'], rnn_model_results['rmse'], xgboost_resutls['rmse'], garch_results['rmse']]
    combined_results['r2'] = [lstm_model_results['r2'], rnn_model_results['r2'], xgboost_resutls['r2'], garch_results['r2']]


    models_performance = pd.DataFrame(combined_results)
    return models_performance

if __name__ == '__main__':
    results = main()
    print(results)
    results.to_csv('../datas/modelperformance.csv', index = False)

### ani tespachi plot the necessary pictures to evaluate the model
