import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd
import torch 
from torch.utils.data import DataLoader, TensorDataset
from src.datasets.sequence_builder import X_train, y_train, X_test, y_test, X_val, y_val 
import random
import numpy as np 

## Setting up the random seed 

RANDOMSEED = 42
random.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(RANDOMSEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
### Creating a generator to pass into the data loader 
g = torch.Generator()
g.manual_seed(RANDOMSEED)

X_train= torch.FloatTensor(X_train)
y_train= torch.FloatTensor(y_train).view(-1,1) ## Adding the dimension to the input

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).view(-1,1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1,1)

### Creating the tensor dataset
train_dataset = TensorDataset(X_train, y_train)
validation_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

### Creating the data loader

train_loader = DataLoader(train_dataset, batch_size=128, shuffle = True, generator=g)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, generator=g)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, generator=g)
