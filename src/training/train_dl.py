import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader
import torch 
import torch.nn as nn
from src.training.trainer_utils import model_train_and_validate

## Setting up the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epochs = 20

## Taking the trainloader, validation loader and test loader to the device
## Setting up the model 
model = RNN(16, 32)
model.to(device)
## Setting up the loss function and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_function = nn.MSELoss()

## Getting the training and validation loss 
train_loss, validation_loss = model_train_and_validate(model, epochs, train_loader, validation_loader, optimizer, loss_function, device)

print(train_loss, validation_loss)

