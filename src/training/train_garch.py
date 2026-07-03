from arch import arch_model 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd 
from src.models.garch import Garch
### Data 
data = pd.read_parquet('./datas/features/combined_data.parquet')

print(data.columns)

all_predictions = []
all_true = []

for ticker, group in data.groupby('ticker'):
    group = group.sort_values('date').reset_index(drop = True)
    train = group[group['date'] < '2022-01-01']
    test = group[group['date'] >= '2022-01-01']
    train_result = train['log_return'].values * 100
    test_result = test['log_return'].values * 100
    true_volataility = test['rolling_vol_5'].values

    ### Fitting the model on the arch model 

    model = arch_model(train_result, vol = 'Garch', p=1, q=1)
    results = model.fit(disp = 'off')

    omega = results.params['omega']
    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']


    ## Forecasting on test 
    garch = Garch(omega, alpha, beta)
    full_returns = 