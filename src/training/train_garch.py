from arch import arch_model 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd 
from src.models.garch import Garch
### Data 
data = pd.read_parquet('./datas/features/combined_data.parquet')


all_predictions = []
all_true = []

for symbol, group in data.groupby('Symbol'):
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
    full_returns = np.concatenate([train_result, test_result])
    full_variance = garch.computing_variance(full_returns)
    test_variance = full_variance[len(train_result):]
    test_vol_pred = np.sqrt(test_variance)


    all_predictions.append(test_vol_pred)
    all_true.extend(true_volataility)

all_predictions = np.array(all_predictions)
all_true = np.array(all_true)

print(all_predictions)
print(all_true)