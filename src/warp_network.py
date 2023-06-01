import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import src.linear_basis as linear_basis
import src.warp as warp
from src.warp_layer import WarpLayer
from src.shift_rows import shift_rows
from src.nan_mean import NanMean
import matplotlib.pyplot as plt

class PerturbationNetwork(nn.Module):
    def __init__(self, n, hidden_size, t):
        super(PerturbationNetwork, self).__init__()
        self.warp = nn.Linear(1, hidden_size, bias=False)
        self.p_out = nn.Linear(hidden_size, 1, bias=False)
        self.relu = nn.ReLU()

  
    def forward(self, x):
        # Create the perturbation profile
        p_i = self.relu(self.warp(x))
        p_i = self.p_out(p_i)

        # Rescale
        p_scale = p_i*p_i.shape[0]

        return p_scale
    

    

class ShiftNetwork(nn.Module):
    def __init__(self, n, hidden_size, t):
        super(ShiftNetwork, self).__init__()
        self.n = n
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.sr = 10
        self.p = None

        # OR this ? Add parameters for the shift path (basis = 5) 
        self.sigma = nn.Parameter(torch.ones(1))
        self.sigma_t = nn.Parameter(torch.ones(1))
        

    def shift_rows(self, X, p):
        # Get the dimensions of the input matrix
        num_rows, num_cols = X.size()

        # Create an empty tensor of the same shape as the input matrix
        shifted_X = torch.torch.full((X.shape[0],X.shape[1]), float('nan'))

        # Iterate over each row of the input matrix
        for i in range(0, (len(p)-1)):
            # Get the shift value for the current row
            shift_amount = int(p[i].item())  # Convert tensor to scalar

            # Perform the row shifting operation
            shifted_X[i*self.sr, :] = torch.roll(X[i*self.sr, :], shifts=shift_amount, dims=0)

        return shifted_X
    
    def nan_mean(self, input):
        # Create a mask for NaN values
        mask = torch.isnan(input)
        
        # Replace NaN values with zeros
        input_zeros = torch.where(mask, torch.zeros_like(input), input)
        
        # Compute the sum and count of non-NaN values along each column
        col_sum = torch.sum(input_zeros, dim=0)
        col_count = torch.sum(~mask, dim=0, dtype=torch.float32)
        
        # Compute the column-wise mean, ignoring NaN values
        output = torch.where(col_count > 0, col_sum / col_count, torch.tensor(float('nan')))
        
        return output
    
    def forward(self, X, p):

        x_shift = self.shift_rows(X, p)
        x_shift  = self.nan_mean(x_shift)

        x_shift = warp.fill_nan_with_last_value(x_shift.squeeze(0))
        x_shift = x_shift[::self.sr]
        x_shift = x_shift[:len(p)]
        x_shift = x_shift.unsqueeze(1)
  
        h1 = self.relu(self.fc1(x_shift))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        y = self.fc4(h3)
        return y.squeeze(1)
    
    def log_likelihood(self, y_pred, y, p):
        
        # Compute log likelihood of data given model parameters
        sigma = torch.exp(self.sigma)
        sigma_t = torch.exp(self.sigma_t)
        dist = Normal(loc=0, scale=sigma)
        ll = dist.log_prob(y_pred-y).sum()
        #print(p)

 
        #print(torch.diff(p, dim=0))

        p_diff = torch.diff(p, dim=0)
    
        #print(p_diff)
        dist2 = Normal(loc=0, scale=sigma_t)
        tl = dist2.log_prob(p_diff.squeeze(0)).sum()

        print(ll)
        print(tl)

        return -ll - tl
    

    
    