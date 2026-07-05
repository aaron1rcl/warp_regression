import torch

# Establish the equation for a Piecewise Linear Spline
def piecewise_linear(x: torch.Tensor, k: float) -> torch.Tensor:
    ''' Linear spline is 0 for all x < k
        and Bx for all x >= k
     For the basis - set B = 1 and let the model learn the parameter'''
    y = x - k
    y = torch.where(x < 0, torch.zeros_like(y), y)
    y = torch.where(x < k, torch.zeros_like(y), y)
    return y

def create_basis(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    ''' Create a linear spline basis'''
    b = torch.empty((len(x), len(k)))
    for i in range(0, len(k)):
        b_i = piecewise_linear(x, k[i])
        b[:,i] = b_i
    return b