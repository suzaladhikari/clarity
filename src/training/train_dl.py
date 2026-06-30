import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader
import torch 
import torch.nn as nn

## Setting up the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epochs = 20

## Setting up the model 
model = RNN(16, 32)
model.to(device)
## Setting up the loss function and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    model.train()
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

    print(f"Epoch {epoch+1}/{epochs}, Loss -> {total_loss/len(train_loader.dataset) :.6f}")


    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_loader: 

            samples, features = batch 
            samples = samples.to(device)
            features = features.to(device)
            output = model(samples)
            loss = loss_function(output, features)
            validation_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss -> {validation_loss/len(validation_loader.dataset) :.6f}")


