import pandas as pd
import os 
import numpy as np

### Here we will be creating new features 

def feature_creation(data):
    ## Creating the log return 
    data['log_return'] = np.log(data['adjClose'] / data['adjClose'].shift(1)) ## Log returns 
    data['squred_log_returns'] = data['log_return'] ** 2
    data['abs_log_returns'] = np.abs(data['log_return'])
    data['ewma_vol_20'] = data['log_return'].ewm(span=20).std()

    numbers = [5,10,20,30]
    for num in numbers:
        column = f'rolling_vol_{num}'
        data[column] = data['log_return'].rolling(num).std() ## Certain window volatility 
    

    data['daily_range'] = (data['adjHigh'] - data['adjLow'])/data['adjClose'] ## Calculating intraday changes/turbulence.

    ## Volume shock 
    volume_mean = data['adjVolume'].rolling(20).mean()
    volume_std = data['adjVolume'].rolling(20).mean()

    data['volume_zscore'] = (data['volume'] - volume_mean) / volume_std

    return data.dropna()