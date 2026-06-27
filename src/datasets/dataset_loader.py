import pandas as pd
import torch 
from torch.utils.data import DataLoader, TensorDataset
from sequence_builder import X_train, y_train, X_test, y_test, X_val, y_val 

X_train= torch.FloatTensor(X_train)
y_train= torch.FloatTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

### Creating the tensor dataset
train_dataset = TensorDataset(X_train, y_train)
validation_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

### Creating the data loader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
