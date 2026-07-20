from arch import arch_model 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd 
from src.models.garch import Garch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json 
data = pd.read_parquet("./datas/combined_data.parquet")

def garch_model(data):
    all_predictions = []
    all_true = []
    symbol_params = {}
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
        symbol_params[symbol] = {"omega": omega, "alpha": alpha, "beta": beta}
        ## Forecasting on test 
        garch = Garch(omega, alpha, beta)
        full_returns = np.concatenate([train_result, test_result])
        full_variance = garch.computing_variance(full_returns)
        test_variance = full_variance[len(train_result):]
        test_vol_pred = np.sqrt(test_variance) /100 


        all_predictions.append(test_vol_pred)
        all_true.extend(true_volataility)

    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)

    all_predictions = all_predictions.flatten()
    all_true = all_true.flatten()

    mae = mean_absolute_error(all_true, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_true, all_predictions))
    r2 = r2_score(all_true, all_predictions)

    path = "./modelperformance/garch_params.json"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(symbol_params, f, indent=4)
    testing_accuracies = {}
    testing_accuracies['model'] = 'garch' 
    testing_accuracies['true_values_garch'] = all_predictions.tolist()
    garch_results = {}
    garch_results['model'] = 'garch'
    garch_results['mae'] = float(mae)
    garch_results['rmse'] = float(rmse)
    garch_results['r2'] = float(r2)

    return garch_results, testing_accuracies
results,testing_accuracies = garch_model(data)
garch_results = results

path = "./modelperformance/performancefiles/garch_results.json"

os.makedirs(os.path.dirname(path), exist_ok=True)

with open(path, "w") as f:
    json.dump(garch_results, f, indent=4)


testing_path = "./modelperformance/garch_predicted.json"
os.makedirs(os.path.dirname(testing_path), exist_ok=True)

with open(testing_path, "w") as f:
    json.dump(testing_accuracies, f, indent=4)