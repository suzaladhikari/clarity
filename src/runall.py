import subprocess, json, os
import sys
import pandas as pd
subprocess.run([sys.executable, 'src/data_pipeline/ingest.py'], check=True)
subprocess.run([sys.executable, 'src/data_pipeline/cleaner.py'], check=True)

subprocess.run([sys.executable, 'src/pipeline.py'], check=True)
subprocess.run([sys.executable, 'src/training/train_xgboost.py'], check=True)
subprocess.run([sys.executable, 'src/training/train_garch.py'], check=True)


json_path = './modelperformance/performancefiles/'
datasets = []
for filename in os.listdir(json_path):
    if filename.endswith('.json'):
        data =os.path.join(json_path, filename)
        
        with open(data, "r") as f:
            data = json.load(f)
        
        datasets.append(data)
flattened_datasets = []

for item in datasets:
    if isinstance(item, list):
        flattened_datasets.extend(item)
    else:
        flattened_datasets.append(item)

combined_datasets = pd.DataFrame(flattened_datasets)
combined_datasets.to_parquet('./datas/modelpeformance.parquet', index = False)