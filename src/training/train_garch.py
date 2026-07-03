from arch import arch_model 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd 
from src.models.garch import Garch
### Data 
data = pd.read_parquet('./datas/features/combined_data.parquet')

print(data.columns)

