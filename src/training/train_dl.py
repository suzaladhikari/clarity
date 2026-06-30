import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader
import torch 
import torch.nn as nn
model = RNN(16, 32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epochs = 20

## Setting up the loss function and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_function = nn.MSELoss()
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero
        samples, features = batch  
        samples = samples.to(device)
        features = features.to(device)

        output,_ = model(samples)
        predicted = output.max(1)