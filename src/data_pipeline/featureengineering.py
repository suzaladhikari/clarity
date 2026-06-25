import pandas as pd
import os 
import numpy as np

### Here we will be creating new features 

def feature_creation(data):
    ## Creating the log return 
    data['return'] = data['adjClose'].pct_change() ## this gives the return 
    data['log_return'] = np.log(data['adjClose'] / data['adjClose'].shift(1)) ## Log returns 
    data['squred_log_returns'] = data['log_return'] ** 2
