import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size,hidden_size, dropout_prob = 0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size = hidden_size, batch_first=True)
        ## So rnn expects the input in terms of (batch, timeStep, features) ## Given the number of batches performs the given number of time step so batch1-> timestep1-> Features/neurons 
    
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out, _  = self.rnn(x) ## This returns the output and the hidden state, _ consists of the final state
        out = self.dropout(out) ## We pass the out to the forward pass and apply the dropout layer
        out = out.permute(0,2,1) ## We then change the structure for batch norm 
        out = self.batch_norm(out) ## We then nomralize through the batch norm 
        out = out.permute(0,2,1) ## Back to the original shape
        return out[:,-1,:] ## Returning the last output of the hidden state
    



