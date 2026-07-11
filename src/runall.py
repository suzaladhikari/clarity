import subprocess, json, os
import sys
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Clarity/src

subprocess.run([sys.executable, os.path.join(BASE_DIR, 'data_pipeline', 'ingest.py')], check=True)
subprocess.run([sys.executable, os.path.join(BASE_DIR, 'pipeline.py')], check=True)
subprocess.run([sys.executable, os.path.join(BASE_DIR, 'training', 'train_xgboost.py')], check=True)
subprocess.run([sys.executable, os.path.join(BASE_DIR, 'training', 'train_garch.py')], check=True)

notebook_path = os.path.join(BASE_DIR, 'modelcomparisons.ipynb')  # adjust if it's actually elsewhere
subprocess.run([
    "jupyter", "nbconvert", "--to", "notebook",
    "--execute", "--inplace",
    notebook_path
], check=True)


json_path = './modelperformance/'
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