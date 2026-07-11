import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.lstm import LSTM
import torch 
import torch.nn as nn
from src.training.trainer_utils import model_train_and_validate, model_test
from src.utils.initials import saving_pickle, loading_pickle

def lstm_model(train_loader, validation_loader, test_loader):
    model = LSTM(16, 32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    epochs = 30
    saving_directory = 'models_saved/lstm/'
    model.to(device)
    ## Setting up the loss function and optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_function = nn.MSELoss()

    ## Getting the training and validation loss 
    calculation= model_train_and_validate(model, epochs, train_loader, validation_loader, optimizer, loss_function, device,saving_directory, 'lstm')
    training_loss = calculation['train_loss']
    validation_loss = calculation['val_loss']
    testing_loss, true_values, predicted, mae, rmse, r2 = model_test(model, test_loader, loss_function, device)
    print("Is it running")
    print(f"Test Loss {testing_loss}, validatiion_loss ->{validation_loss[-1]}, t{training_loss[-1]}, MAE:{mae}, RMSE:{rmse}, r2:{r2}")


    lstm_results = {}
    lstm_results['true_values'] = true_values
    lstm_results['predicted'] = predicted
    lstm_results['mae'] = mae
    lstm_results['rmse'] = rmse
    lstm_results['r2'] = r2
    return lstm_results

