from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix, recall_score, accuracy_score, ConfusionMatrixDisplay,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from models.xgboost import build_xgboost_model
import pandas as pd
import numpy as np

data = pd.read_parquet('./datas/features/combined_data.parquet')

