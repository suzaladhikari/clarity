import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_garch import params_garch
from src.models.garch import Garch
import pandas as pd

def return_value_by_ticker(ticker, datapath = './datas/combined_data.parquet'):
     data = pd.read_parquet(datapath)
     ticker_data = data[data['Symbol'] == ticker].sort_values(by = 'date').reset_index(drop = True)
     if ticker_data.empty:
        raise ValueError(f"There is no data of symbol : {ticker}")
     returns = ticker_data['log_return'].values * 100 
     return returns 