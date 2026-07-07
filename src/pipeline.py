## Creating the pipeline so that I donot have to rerun the code over again and again 
## Necessary imports 
import sys 
import os 
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_pipeline.ingest import watchlist ## Function to extract the company dataset 
from src.data_pipeline.cleaner import data_corrector ## to correct and check the nans and infs
from src.data_pipeline.featureengineering import feature_creation ## Creating new features
def extracting_to_final_data(key):
    correctedData = data_corrector(key)
    feature_engineering = feature_creation(correctedData)
    return feature_engineering

combined_data = []
for ticker in watchlist.keys():
    stock_data = extracting_to_final_data(ticker)
    combined_data.append(stock_data)

combined_stock_data = pd.concat(combined_data, ignore_index=True)

watchlist = {
    "AAPL": "Apple", 
    "MSFT": "Microsoft"
}
## Building the sequences

### Pipeline ma chai aaba seqeucne ani dataloader ko finalize garne 
## Ani tes pachi make sure you save the whole files in the dictionary ! got it 
## Ani __name__ == __main__ ko use ni k huncha bhanera bujne 
## Ani make sure you create the reusable function in seqeucen builder and also the dataloader to carry on your project