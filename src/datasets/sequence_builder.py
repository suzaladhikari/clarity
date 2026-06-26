# This .py file makes sequenes ready for the RNN and LSTM files 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import os 
## Creating a combined datset 

combined_directory = '/datas/features'
combined_data = []
for filename in os.listdir(combined_directory):
    if filename.endswith('.parquet'):
        file_path = os.path.join(combined_directory, filename)
        data = pd.read_parquet(file_path)
        combined_data.append(data)
    
combined_stock_data = pd.concat(combined_data, ignore_index=True)
print(combined_stock_data.shape[0])