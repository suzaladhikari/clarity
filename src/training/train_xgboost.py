import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.xgboost import build_xgboost_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix, recall_score, accuracy_score, ConfusionMatrixDisplay,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

data = pd.read_parquet('./datas/features/combined_data.parquet')

# Train,test and evaluation split

train_split = data[data['date'] < '2020-01-01'].copy()
validation_split = data[(data['date'] >= '2020-01-01' ) & (data['date'] < '2022-01-01')].copy()
test_split = data[data['date'] >= '2022-01-01'].copy()

