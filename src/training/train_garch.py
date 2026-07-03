from arch import arch_model 
import numpy as np
import pandas as pd 
### Data 
data = pd.read_csv('/datas/features/combined_data.parquet')

training_data = data[data['date'] < '2022-01-01'].copy()
testing_data = data[data['date'] >= '2022-01-01'].copy()

model = arch_model(training_data['log_returns'])

print(data['log_returns'])