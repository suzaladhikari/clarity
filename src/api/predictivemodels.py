import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_garch import params_garch
from src.models.garch import Garch
import pandas as pd


def return_value_by_ticker(ticker, datapath = './datas/combined_data.parquet'):
     ticker_params = params_garch[ticker]
     omega, alpha, beta = ticker_params['omega'], ticker_params['alpha'], ticker_params['beta']
     data = pd.read_parquet(datapath)
     ticker_data = data[data['Symbol'] == ticker].sort_values(by = 'date').reset_index(drop = True)
     if ticker_data.empty:
        raise ValueError(f"There is no data of symbol : {ticker}")
     returns = ticker_data['log_return'].values * 100 
     return returns, omega, alpha, beta

def predict_next_day_volatility(returns, omega, alpha, beta):
    model = Garch(omega, alpha, beta)
    variance = model.computing_variance(returns)
    next_day = model.forecast_next_day(returns, variance)

    return next_day
returns, omega, alpha, beta = return_value_by_ticker("NVIDIA")
next_day = predict_next_day_volatility(returns, omega, alpha, beta)
print(next_day)
