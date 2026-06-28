import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.xgboost import build_xgboost_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix, recall_score, accuracy_score, ConfusionMatrixDisplay,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from src.datasets.sequence_builder import train_split, test_split, validation_split


# Train,test and evaluation split

print(train_split.shape[0], test_split.shape[0], validation_split.shape[0])
