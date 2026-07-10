import subprocess, json, os
import sys
subprocess.run([sys.executable, 'src/pipeline.py'], check=True)
subprocess.run([sys.executable, 'src/training/train_xgboost.py'], check =True)
subprocess.run([sys.executable, 'src/training/train_garch.py'], check =True)
