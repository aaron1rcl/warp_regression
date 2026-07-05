import torch
import numpy as np
import torch.nn.functional as F
from xitorch.interpolate import Interp1D

def matrix_decomp(x: torch.Tensor, n: int, sr: int):

    if x.dim() == 2:
        x = x[:,0]

    x_i = torch.full((n*sr,), float('nan'), dtype=torch.float32)
    pos = torch.arange(0, n)*sr
    x_i[pos] = x
    ones = torch.ones(x_i.shape[0], dtype=torch.float32)
    # Create 1.T
    ones = ones.unsqueeze(0)
    x_i = x_i.unsqueeze(1)
    X = torch.mm(x_i, ones)
    I = torch.eye(n*sr)
    I[I == 0] = float('nan')
    X = X*I
    return X

def shift(x: torch.Tensor, k=1):
    ''' shift the arrays'''
    if k > 0:
        y = torch.cat([torch.full((k,), float('nan')), x])
        y = y[:len(x)]
    elif k < 0:
        k = abs(k)
        y = torch.cat([x, torch.full((k,), float('nan'))])
        y = y[k:]
    else:
        y = x
    return y

def X_shift(X: torch.Tensor, sr: int, p: torch.Tensor):
    # First enlarge the array to include the largest positive time shift
    if torch.max(p) > 0:
        X_out = F.pad(X.clone().detach(), (0, X.shape[0]), mode='constant', value=float('nan'))
    else:
        X_out = X.clone().detach()

    # Shift the values
    for i in range(1, int(X.shape[0]/sr)):
        x_r = X_out[i*sr, :]
        x_shift = shift(x_r, int(p[i].item()))
        X_out[i*sr, :] = x_shift

    return X_out

def fill_nan_with_last_value(x):
    mask = torch.isnan(x)

    last = x[0].item()
    for i in range(1, x.shape[0]):
        if mask[i].item():
            x[i] = last
        elif mask[i].item() is False:
            last = x[i].item()
    return x




