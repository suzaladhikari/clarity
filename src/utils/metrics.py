import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred )

    return y_pred, rmse, mae,r2

## For the deep learning models 

def deep_evaluation(predicted, true_values):
    predicted = torch.cat(predicted)
    true_values = torch.cat(true_values)
    mae = torch.mean(torch.abs(predicted - true_values)).item()
    rmse = torch.sqrt(torch.mean((predicted - true_values) ** 2)).item()
    ss_res = torch.sum((true_values - predicted) ** 2)
    ss_tot = torch.sum((true_values - true_values.mean()) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    return mae, rmse, r2