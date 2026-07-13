import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_garch import params_garch
from src.models.garch import Garch
import pandas as pd
import numpy as np 
import xgboost as xgb 
### GARCH MODEL 
class GarchModel: 
    def __init__(self, ticker):
        self.ticker = ticker.upper()
    def return_value_by_ticker(self, datapath = './datas/combined_data.parquet'):
        ticker_params = params_garch[self.ticker]
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
        self.ticker = ticker
        self.model = xgb.XGBRegressor()
        self.model.load_model('./models_saved/xgboost/xgboost_model.json')
    
    def latest_data(self):
        data = pd.read_parquet('./datas/combined_data.parquet')
        ticker_data = data[data['Symbol'] == self.ticker].sort_values(by = "date")
        if ticker_data.empty:
            raise ValueError(f"There is no data of symbol : {self.ticker}")    
        ## Extracting the last column of the data 
        latest_row = ticker_data.iloc[-1:]
        DROP_COLS = ['date', 'Symbol', 'target_volatility']
        X = latest_row.drop(columns=DROP_COLS)
        return X
    
    def predict(self):
        x = self.latest_data()
        pred = self.model.predict(x)
        return pred[0]
    


def predict_volatility(ticker:str, model:str):
    if model == 'garch':
        return GarchModel(ticker).predict_next_day_volatility()
    
