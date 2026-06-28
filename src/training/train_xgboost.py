import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.xgboost import build_xgboost_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


combined_stock_data = pd.read_parquet('./datas/features/combined_data.parquet')
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
