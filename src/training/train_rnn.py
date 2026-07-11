import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
import torch 
import torch.nn as nn
from src.utils.initials import saving_pickle, loading_pickle
from src.training.trainer_utils import model_train_and_validate, model_test


def rnn_model(train_loader, validation_loader, test_loader):
    ## Setting up the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    epochs = 25
    saving_directory = 'models_saved/rnn/'
    checkpoint_dir = f"{saving_directory}rnn_best.pt"
    ## Taking the trainloader, validation loader and test loader to the device
    ## Setting up the model 
    model = RNN(16, 32)
## Loading the saved directory only if it exists
    model.to(device)
    ## Setting up the loss function and optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_function = nn.MSELoss()

    ## Getting the training and validation loss 
    calculation= model_train_and_validate(model, epochs, train_loader, validation_loader, optimizer, loss_function, device,saving_directory, 'rnn')
    training_loss = calculation['train_loss']
    validation_loss = calculation['val_loss']
    testing_loss, true_values, predicted, mae, rmse, r2 = model_test(model, test_loader, loss_function, device)

    print("Is it running")
    print(f"Test Loss {testing_loss}, validatiion_loss ->{validation_loss[-1]}, t{training_loss[-1]}, MAE:{mae}, RMSE:{rmse}, r2:{r2}")

    rnn_results = {}
    rnn_results['true_values'] = true_values
    rnn_results['predicted'] = predicted
    rnn_results['mae'] = mae
    rnn_results['rmse'] = rmse
    rnn_results['r2'] = r2

    return rnn_results



