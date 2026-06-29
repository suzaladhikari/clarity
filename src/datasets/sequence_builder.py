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
os.makedirs('./datas/features', exist_ok=True)


## Dropping the unnecessary columns 
columns_to_drop = ['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh','adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor',]
combined_stock_data = combined_stock_data.drop(columns = columns_to_drop)
combined_stock_data.to_parquet('./datas/features/combined_data.parquet' , index = False)
## Doing the train, validation and test split with the date
train_split = combined_stock_data[combined_stock_data['date'] < '2020-01-01'].copy()
validation_split = combined_stock_data[(combined_stock_data['date'] >= '2020-01-01' ) & (combined_stock_data['date'] < '2022-01-01')].copy()
test_split = combined_stock_data[combined_stock_data['date'] >= '2022-01-01'].copy()

## Scaling the features

scaler = StandardScaler()
scalable_columns = combined_stock_data.drop(columns = ['date', 'Symbol', 'target_volatility']).columns
train_split[scalable_columns] = scaler.fit_transform(train_split[scalable_columns])
validation_split[scalable_columns] = scaler.transform(validation_split[scalable_columns])
test_split[scalable_columns] = scaler.transform(test_split[scalable_columns])


## Creating Sequences 
def creating_sequences(data, scalable_columns, target_col = 'target_volatility', window = 30):
    X,y = [], []
    ### For each stock 
    for symbol in data['Symbol'].unique():
        stock = data[data['Symbol'] == symbol].sort_values('date') ## Sorting values by date 
        features = stock[scalable_columns].values
        target = stock[target_col].values
        for i in range(window, len(stock)): ## This creates the slide window like 0-30, 1-31, 2-32
            X.append(features[i-window:i]) ## This appends the features
            y.append(target[i]) ## This appends the target for the 30+i days 
    
    return np.array(X), np.array(y)

## The shape of X is : (4586,30,16) ## So there are  4586 samples, 30 window, and 16 features 
## Similarly y is (4586, ) meaning only 4586 values 
X_train, y_train = creating_sequences(train_split, scalable_columns)
X_val, y_val = creating_sequences(validation_split, scalable_columns)
X_test, y_test = creating_sequences(test_split, scalable_columns)

print(X_train.shape)