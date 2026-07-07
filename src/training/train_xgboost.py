import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.xgboost import build_xgboost_model, saving_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.metrics import evaluate_model

def xgboost_model(combined_stock_data):
## Doing the train, validation and test split with the date
    train_split = combined_stock_data[combined_stock_data['date'] < '2020-01-01'].copy()
    validation_split = combined_stock_data[(combined_stock_data['date'] >= '2020-01-01' ) & (combined_stock_data['date'] < '2022-01-01')].copy()
    test_split = combined_stock_data[combined_stock_data['date'] >= '2022-01-01'].copy()
    DROP_COLS = ['date', 'Symbol', 'target_volatility']
    TARGET = 'target_volatility'
    ## Scaling the features

    scaler = StandardScaler()
    scalable_columns = [c for c in combined_stock_data.columns if c not in DROP_COLS]
    train_split[scalable_columns] = scaler.fit_transform(train_split[scalable_columns])
    validation_split[scalable_columns] = scaler.transform(validation_split[scalable_columns])
    test_split[scalable_columns] = scaler.transform(test_split[scalable_columns])

    X_train = train_split[scalable_columns] 
    y_train = train_split[TARGET]

    X_val = validation_split[scalable_columns]
    y_val = validation_split[TARGET]

    X_test = test_split[scalable_columns]
    y_test = test_split[TARGET]

    ## Loading data is done ! 

    default_params = {
        'n_estimators': 570,  ## Setting it high and using early stopping rounds to determine what to choose     
        'max_depth': 4,             
        'learning_rate': 0.01,      
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,      
        'gamma': 0.1,               
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    saving_model(model)
    model = build_xgboost_model(default_params) # Built the XGBoost Regressor with the given parameters
    model.fit(X_train,y_train, eval_set = [(X_val, y_val)], verbose = 30)
    rmse, mae,r2 = evaluate_model(model, X_test, y_test)

    xgboost_results = {}
    xgboost_results['mae'] = mae
    xgboost_results['rmse'] = rmse
    xgboost_results['r2'] = r2

    return xgboost_results

