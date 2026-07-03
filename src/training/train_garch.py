from arch import arch_model 
import numpy as np
import pandas as pd 
### Data 
data = pd.read_parquet('./datas/features/combined_data.parquet')

training_data = data[data['date'] < '2022-01-01'].copy()
testing_data = data[data['date'] >= '2022-01-01'].copy()

model = arch_model(training_data['log_return'])
results = model.fit(disp = 'off') ## Used this to get the cleaner output

print(results.params)