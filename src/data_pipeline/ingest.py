### Libraries to be imported
import pandas as pd 
import numpy as np 
import os 
import time
import requests
import glob
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# for f in glob.glob('../updatedtop20datasets/*'):
#     os.remove(f)
#     print("Deleted", f)
    
### Creating datasets for the top 20 companies 
watchlist = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft", 
    "NVDA":  "Nvidia",
    "GOOGL": "Alphabet (Google)",
    "AMD":  "AMD",
    "META":  "Meta (Facebook)",
    "TSLA":  "Tesla",
    "NFLX": "Netflix",
    "LLY":   "Eli Lilly",
    "AVGO":   "Broadcom",
    "MU":    "Micron Technology",
    "QCOM":   "Qualcomm",
    "UNH":   "UnitedHealth",
    "WMT":   "Walmart",
    "MA":    "Mastercard",
    "JNJ":   "Johnson & Johnson",
    "PG":    "Procter & Gamble",
    "HD":    "Home Depot",
    "ORCL":  "Oracle",
    "JPM": "JPMorgan Chase"
}

def company_datasets(symbol):
    path = os.path.join("datas", "raw")
    os.makedirs(path,exist_ok=True)
    filepath = os.path.join(path, f"{symbol}.parquet")
    if os.path.exists(filepath):
        existing_data = pd.read_parquet(filepath)
        if "Symbol" not in existing_data.columns:
            existing_data["Symbol"] = symbol
        last_date = pd.to_datetime(existing_data['date']).max()
        start_date = (last_date + pd.Timedelta(days = 1)).strftime('%Y-%m-%d')

        if pd.to_datetime(start_date) > pd.Timestamp.today():
            return existing_data
        API_KEY = '8e14d5babfe29c8815f268eb1afa1727ce18f16e'
        url_link = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        headers = {'Content-Type':'application/json'}
        params = {'startDate': start_date, 'token':API_KEY}
        response = requests.get(url_link, headers=headers, params=params)
        new_data = pd.DataFrame(response.json())
        if new_data.empty:
            return existing_data
        
        new_data["Symbol"] = symbol
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep = 'last')
        combined.to_parquet(filepath, index = False)
        return combined
    
    API_KEY = '8e14d5babfe29c8815f268eb1afa1727ce18f16e'
    url_link = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    headers = {'Content-Type':'application/json'}
    params = {'startDate': '2008-01-01', 'token':API_KEY}
    respone = requests.get(url_link, headers = headers, params = params)
    data = respone.json()
    data = pd.DataFrame(data)
    if data.empty:
        return data 
    data["Symbol"] = symbol
    data.to_parquet(filepath, index = False)
    return data



### Market tracking 
market = {'SPY': 'S & P 500'}
if __name__ == "__main__":
    for key, value in watchlist.items():
        company_datasets(key)
    for key in market.keys():
        company_datasets(key)




