import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.25)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def init_hidden(self):
        # Initializes hidden layer with zeros
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, input_X):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        # Input should be reshaped from [batch_size] to [1, batch_size, input_dim]
        lstm_input = input_X.view(1, self.batch_size, self.input_dim)
        lstm_out, _ = self.lstm(lstm_input.float())
        out = self.linear(lstm_out)
        
        return out
