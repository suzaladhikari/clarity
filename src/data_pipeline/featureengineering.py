import pandas as pd
import os 
import numpy as np

### Here we will be creating new features 

### S & P 500 
sp5 = pd.read_parquet('/Users/sujaladhikari/sujalpersonal/Projects/Clarity/datas/processed/SPY.parquet')

def feature_creation(data):
    ## Creating the log return 
    data['log_return'] = np.log(data['adjClose'] / data['adjClose'].shift(1)) ## Log returns 
    data['squared_log_returns'] = data['log_return'] ** 2
    data['abs_log_returns'] = np.abs(data['log_return'])
    data['ewma_vol_20'] = data['log_return'].ewm(span=20).std()

    numbers = [5,10,20,30]
    for num in numbers:
        column = f'rolling_vol_{num}'
        data[column] = data['log_return'].rolling(num).std() ## Certain window volatility 
    

    data['daily_range'] = (data['adjHigh'] - data['adjLow'])/data['adjClose'] ## Calculating intraday changes/turbulence.

    ## Volume shock 
    volume_mean = data['adjVolume'].rolling(20).mean()
    volume_std = data['adjVolume'].rolling(20).std()

    data['volume_zscore'] = (data['adjVolume'] - volume_mean) / volume_std
    data['vol_lag_1'] = data['rolling_vol_20'].shift(1)

    return data.dropna()