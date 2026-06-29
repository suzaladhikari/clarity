import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader
import torch 
model = RNN(16, 32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epochs = 20
for epoch in range(epochs):
    for batch in train_loader:
        samples, features = batch  
        samples = samples.to(device)
        features = features.to(device)
