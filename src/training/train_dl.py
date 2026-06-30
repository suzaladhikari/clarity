import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader
import torch 
import torch.nn as nn



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epochs = 20

## Setting up the model 
model = RNN(16, 32)
model.to(device)
## Setting up the loss function and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_function = nn.MSELoss()
model.train()
for epoch in range(epochs):
    total_loss = 0 
    for batch in train_loader:
        optimizer.zero_grad()
        samples, features = batch  
        samples = samples.to(device)
        features = features.to(device)
        output = model(samples)
        loss = loss_function(output, features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

