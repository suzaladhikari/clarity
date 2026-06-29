import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.rnn import RNN
from src.datasets.dataset_loader import train_loader, validation_loader, test_loader

print(train_loader)