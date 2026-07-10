import subprocess, json, os
import sys
import pandas as pd
# subprocess.run([sys.executable, 'src/pipeline.py'], check=True)
# subprocess.run([sys.executable, 'src/training/train_xgboost.py'], check =True)
# subprocess.run([sys.executable, 'src/training/train_garch.py'], check =True)

json_path = './modelperformance/'
datasets = []
for filename in os.listdir(json_path):
    if filename.endswith('.json'):
        data =os.path.join(json_path, filename)
        data = pd.read_json(data)
        datasets.append(data)

combined_datasets = pd.concat(datasets, ignore_index=True)
combined_datasets.head(5)