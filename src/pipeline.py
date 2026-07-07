## Creating the pipeline so that I donot have to rerun the code over again and again 
## Necessary imports 

import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_pipeline.ingest import company_datasets ## Function to extract the company dataset 


