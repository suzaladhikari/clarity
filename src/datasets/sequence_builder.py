# This .py file makes sequenes ready for the RNN and LSTM files 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os 
## Creating a combined datset 

combined_directory = './datas/features'
combined_data = []
for filename in os.listdir(combined_directory):
    clean_name = filename.strip().strip("'").strip('"')
    if clean_name.endswith('.parquet'):
        file_path = os.path.join(combined_directory, filename)
        data = pd.read_parquet(file_path)
        combined_data.append(data)
    
combined_stock_data = pd.concat(combined_data, ignore_index=True)
print(combined_stock_data.shape[0])

## Dropping the unnecessary columns 
columns_to_drop = ['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh','adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor',]
combined_stock_data = combined_stock_data.drop(columns = columns_to_drop)

## Doing the train, validation and test split with the date
train_split = combined_stock_data[combined_stock_data['date'] < '2020-01-01'].copy()
validation_split = combined_stock_data[(combined_stock_data['date'] >= '2020-01-01' ) & (combined_stock_data['date'] < '2022-01-01')].copy()
test_split = combined_stock_data[combined_stock_data['date'] >= '2022-01-01'].copy()
print(train_split.shape, validation_split.shape, test_split.shape)

## Scaling the features

scaler = StandardScaler()
scalable_columns = combined_stock_data.drop(columns = ['date', 'Symbol', 'target_volatility']).columns
train_split[scalable_columns] = scaler.fit_transform(train_split[scalable_columns])
validation_split[scalable_columns] = scaler.transform(validation_split[scalable_columns])
test_split[scalable_columns] = scaler.transform(test_split[scalable_columns])