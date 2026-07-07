## Creating the pipeline so that I donot have to rerun the code over again and again 
## Necessary imports 

import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_pipeline.ingest import company_datasets ## Function to extract the company dataset 
from src.data_pipeline.cleaner import data_corrector ## to correct and check the nans and infs
from src.data_pipeline.featureengineering import feature_creation ## Creating new features
def extracting_to_final_data(key):
    rawdata = company_datasets(key)
    correctedData = data_corrector(rawdata)
    feature_engineering = feature_creation(correctedData)
    return feature_engineering

