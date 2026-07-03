from arch import arch_model 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd 
from src.models.garch import Garch
### Data 
data = pd.read_parquet('./datas/features/combined_data.parquet')

training_data = data[data['date'] < '2022-01-01'].copy()
testing_data = data[data['date'] >= '2022-01-01'].copy()

model = arch_model(training_data['log_return'] * 100, vol = 'Garch', p = 1, q = 1)
results = model.fit(disp = 'off') ## Used this to get the cleaner output

### Extracting the omega, alpha and beta
omega = results.params['omega']
alpha = results.params['alpha[1]']
beta = results.params['beta[1]']

## Garch model 
garch = Garch(omega, alpha, beta )

print(garch )