import pandas as pd
import os 
import numpy as np
from ingest import watchlist
### Here we will be creating new features 

### S & P 500 
sp5 = pd.read_parquet('/Users/sujaladhikari/sujalpersonal/Projects/Clarity/datas/processed/SPY.parquet')
sp5.index = pd.to_datetime(sp5.index).tz_localize(None)
sp5['sp5_log_return'] = np.log(sp5['adjClose'] / sp5['adjClose'].shift(1))
sp5['sp5_20_volatility'] = sp5['sp5_log_return'].rolling(20).std()
sp5['sp5_abs_return'] = np.abs(sp5['sp5_log_return'])
sp5 = sp5.sort_index()

def feature_creation(data):
    ## Creating the log return 
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    data = data.merge(sp5[['sp5_log_return', 'sp5_20_volatility', 'sp5_abs_return']], left_on = 'date', right_index = True, how= 'left')
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

    ### Linkage with S and P 500 
    data['excess_return'] = data['log_return'] - data['sp5_log_return']
    data['vol_ratio'] = data['rolling_vol_20'] / data['sp5_20_volatility']

    return data.dropna()

for key in watchlist.keys():
    filepath = os.path.join("datas", "processed")
    filename = os.path.join(filepath, f"{key}.parquet")
    savingpath = os.path.join("datas", "features", f"{key}.parquet")
    data = pd.read_parquet(filename)
    processed_data = feature_creation(data)
    processed_data.to_parquet(savingpath, index=False)

