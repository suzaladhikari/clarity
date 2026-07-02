import torch.nn as nn 
import torch 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_state, dropout_prob = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_state, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_state)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_prob),
            nn.Linear(64,32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_prob),
            nn.Linear(32,16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_prob),
            nn.Linear(16,1)
        )
