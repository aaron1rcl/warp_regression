import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class PerturbationNetwork(nn.Module):
    def __init__(self, n_knots, t):
        super(PerturbationNetwork, self).__init__()
        self.warp = nn.Parameter(n_knots)
        self.t = t
        self.n_knots = 5
        self.knots = torch.linspace(0, t, steps=n_knots)
        self.X = self.basix_matrix()
        self.rw_interval = torch.diff(self.k_values)[0].item()

    def basix_matrix(self):
        x = torch.arange(1, self.t + 1)  # Sample x values
        self.k_values = np.arange(0, self.n_knots) * (self.t / self.n_knots)  # Values of k
        X = self.create_basis(x, self.k_values)
        return X/torch.max(X)

    def piecewise_linear(x, k):
        y = x - k
        y[x < 0] = 0
        y[x < k] = 0
        return y

    def create_basis(self, x, k):
      num_x = len(x)
      num_k = len(k)
      b = torch.zeros((num_x, num_k))
      for i in range(num_k):
          b_i = self.piecewise_linear(x, k[i])
          b[:, i] = b_i
      return b

    def forward(self):
        # Create the perturbation profile and return
        p = np.dot(self.X, self.B)
        return p
    
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

    def fill_nan_with_last_value(x):
        mask = torch.isnan(x)

        last = x[0].item()
        for i in range(1, x.shape[0]):
            if mask[i].item():
                x[i] = last
            elif mask[i].item() is False:
                last = x[i].item()
        return x

    def conditional_posterior(d, u, sd, n_steps):
        ''' Posterior Conditional conditional on f(X | Xn = d) Distribution
        Multivariate gaussian distribution'''
        u = (d / n_steps)
        sd = torch.sqrt((sd ** 2) * (n_steps - 1) / n_steps)
        return u, sd

    def px_z(u1, u2, sd1, sd2):
        ''' Find the pdf of Z ~N(u2, sd2**2), given the parameters of X ~ (u1, sd1**2) '''

        # Define the constants
        alpha = torch.sqrt(1 / (2 * sd2**2) + 1 / (2 * sd1**2))
        alpha_beta = u2 / (2 * sd2**2) + u1 / (2 * sd1**2)
        gamma = (u2**2) / (2 * sd2**2) + (u1**2) / (2 * sd1**2)
        beta = alpha_beta / alpha

        # Define the components
        p1 = 1 / (2 * sd1 * sd2 * alpha * torch.sqrt(torch.tensor(np.pi)))
        p2 = torch.exp(-gamma + beta**2)

        return p1 * p2

    def reverse_softplus(x):
        return -torch.log(1 + torch.exp(-x))

    def expected_likelihood(self, d, u, sd, n_steps):
        ''' Analytical Solution to the Expected Likelihood
        Given a random walk what is the expected likelihood
        of the walk, given that it took n_steps and moved a vertical distance d'''
        post_u, post_sd = self.conditional_posteriorposterior(d, u, sd, n_steps)
        el = torch.log(self.px_z(u1=torch.tensor(0.0), u2=post_u, sd1=sd, sd2=post_sd))
        # Apply a reversed softplus so that the values are bounded to P=1
        el = self.reverse_softplus(el * n_steps)
        return el

    def terror_likelihood(self, k_values, sd, RW_width, p):
        ''' Calculate the likelihood across on the time axis 
        The returned value comprises the expected likelihood, calculated
        between each of the k knot values'''
        total_likelihood = torch.tensor(0.0)
        for i in range(1, len(k_values)):
          x1 = k_values[i]
          x2 = k_values[i-1]

          y1 = p[x1]
          y2 = p[x2]

          d = y2 - y1

          el = self.expected_likelihood(d, 0, sd, RW_width)
          total_likelihood =+ el
        return total_likelihood
    
    def forward(self, X, p):

        x_shift = self.shift_rows(X, p)
        x_shift  = self.nan_mean(x_shift)

        x_shift = self.fill_nan_with_last_value(x_shift.squeeze(0))
        x_shift = x_shift[::self.sr]
        x_shift = x_shift[:len(p)]
        x_shift = x_shift.unsqueeze(1)
  
        h1 = self.relu(self.fc1(x_shift))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        y = self.fc4(h3)
        return y.squeeze(1)
    
    def log_likelihood(self, y_pred, y, p, k_values, rw_interval):
        
        # Compute log likelihood of data given model parameters
        sigma = torch.exp(self.sigma)
        sigma_t = torch.exp(self.sigma_t)
        dist = Normal(loc=0, scale=sigma)
        ll = dist.log_prob(y_pred-y).sum()
 
        tl = self.terror_likelihood(k_values, sigma_t, rw_interval, p)
    
        # Maximize therefore multiply by -1
        return -(ll + tl)
    

    
    
