import pandas as pd
import os 
import numpy as np

### Here we will be creating new features 

def feature_creation(data):
    ## Creating the log return 
    data['return'] = data['adjClose'].pct_change() ## this gives the return 
    data['log_return'] = np.log(data['adjClose'] / data['adjClose'].shift(1)) ## Log returns 
    data['squred_log_returns'] = data['log_return'] ** 2

# Calculating the rolling volatitlity 
    data['rolling_vol_5'] = data['log_return'].rolling(5).std()
    data['rolling_vol_30'] = data['log_return'].rolling(30).std()
    