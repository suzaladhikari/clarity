import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.garch import Garch
from src.data_pipeline.featureengineering import feature_creation
from src.datasets.sequence_builder import train_test_val
import pandas as pd
import numpy as np 
import xgboost as xgb 
import json 
import joblib
import torch.nn as nn 
import torch 
### GARCH MODEL 
class GarchModel: 
    def __init__(self, ticker):
        self.ticker = ticker.upper()
    def return_value_by_ticker(self, datapath = './datas/combined_data.parquet'):
        with open("./modelperformance/garch_params.json", "r") as f: 
            data = json.load(f)
        ticker_params = data[self.ticker]
        omega, alpha, beta = ticker_params['omega'], ticker_params['alpha'], ticker_params['beta']
        data = pd.read_parquet(datapath)
        ticker_data = data[data['Symbol'] == self.ticker].sort_values(by = 'date').reset_index(drop = True)
        if ticker_data.empty:
            raise ValueError(f"There is no data of symbol : {self.ticker}")
        returns = ticker_data['log_return'].values * 100 
        return returns, omega, alpha, beta

    def predict_next_day_volatility(self):
        returns, omega, alpha, beta = self.return_value_by_ticker()
        model = Garch(omega, alpha, beta)
        variance = model.computing_variance(returns)
        next_day = model.forecast_next_day(returns, variance)
        next_day = np.sqrt(next_day) / 100
        return next_day

### XGBoost Model 
class XGBoostModel: 
    def __init__(self, ticker):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        model_path = os.path.join(
            base_dir,
            "models_saved",
            "xgboost",
            "xgboost_model.json"
        )
        self.ticker = ticker.upper()
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
    
    def latest_data(self):

        path = f'./datas/processed/{self.ticker}.parquet'
        data = pd.read_parquet(path)
        featured_data = feature_creation(data, training=False)
        if featured_data.empty:
            raise ValueError(f"There is no data of symbol : {self.ticker}")  
        print(featured_data['date'].max())  
        ## Extracting the last column of the data 
        latest_row = featured_data.iloc[-1:]
        DROP_COLS = ['date', 'Symbol']
        X = latest_row.drop(columns=DROP_COLS)
        return X
    
    def predict(self):
        x = self.latest_data()
        pred = self.model.predict(x)
        return pred[0]
    

class SequenceCreater: 
    def __init__(self, ticker, scaler):
        self.ticker = ticker 
        self.scaler = scaler
        self.window = 30
        self.data = pd.read_parquet(f'./datas/processed/{self.ticker}.parquet')
    def creating_sequences(self):
        featured_data = feature_creation(self.data, training=False)
        columns_to_drop = ['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh','adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor','date', 'Symbol']
        featured_data = featured_data.drop(columns = columns_to_drop)
        columns = featured_data.columns
        featured_data[columns] = scaler.transform(featured_data[columns])
        latest_sequence = featured_data[columns].tail(30).values
        X = np.expand_dims(latest_sequence, axis = 0)
        return X 





scaler = joblib.load('./models_saved/scaler.pkl')

model = SequenceCreater("AAPL", scaler).creating_sequences()

### Creating the RNN model 
def rnn(state_dict_path):
    model = nn.RNN(input_size=16,hidden_size=64)
    model_dict = torch.load(state_dict_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model
### Based on the user's request 

def predict_volatility(ticker:str, model:str):
    if model == 'garch':
        return GarchModel(ticker).predict_next_day_volatility()
    if model == 'xgboost':
        return float(XGBoostModel(ticker).predict())
    if model == 'rnn':
        model = rnn('./models_saved/rnn/rnn_best.pt')
        X = SequenceCreater(ticker).creating_sequences()
        with torch.no_grad():
            prediction = model(X)
        return prediction

