## Handling all the missing values 

import os 
import pandas as pd 
import numpy as np 
from src.data_pipeline.ingest import watchlist, market

def data_corrector(symbol): ## Data Corrector function ot detect invalid and nan data types
    path = os.path.join("datas", "raw")
    saving_path = os.path.join("datas", "processed")
    filepath = os.path.join(path, f"{symbol}.parquet")
    savingname = os.path.join(saving_path, f"{symbol}.parquet")
    if not os.path.exists(filepath): ## If there is no such file
        print(f"There is no such file for the symbol {symbol}")
        return None ## Returning none
    data = pd.read_parquet(filepath) ## Reading the paraquet
    data = data.sort_values(by = 'date')
    data = data.loc[:, ~data.columns.str.contains('Unnamed')] ### removing the unnamed column
    data['date'] = pd.to_datetime(data['date']) ## Converting the date to pd.datetime format
    if data.isna().sum().sum() > 0: ## Checking the nan values
        print(f"Missing Values Detected for symbol {symbol}, Fixed using forward fill")
        data = data.ffill() ## Doing forward fill for all the missing values 
    if data.duplicated(subset=['date']).sum() > 0: ## If there are duplicates 
        data = data.drop_duplicates(subset=['date']) ## Dropping htem 
        print(f"There are duplicate values for stock {symbol}. Fixed using duplication drop")
    data.to_parquet(savingname, index = False) ## Converting them back to parquet and stroing them in the processed 
    return data

if __name__ == "__main__":
    for key in watchlist.keys():
        data_corrector(key)
    for key in market.keys():
        data_corrector(key)
