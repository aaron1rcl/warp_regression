import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from patsy import dmatrix


class ShiftNetwork(nn.Module):
    def __init__(self, n, hidden_size, t, n_knots, opt_max=False):
        super(ShiftNetwork, self).__init__()

        # Settings
        self.n = n
        self.hidden_size = hidden_size
        self.n_knots = n_knots
        self.t = t
        self.sr = 10
        self.scale_factor=0.02
        self.opt_max = opt_max

        # Trainable Parameters 
        self.B = nn.Parameter(torch.randn(n_knots))

        self.fc1 = nn.Linear(n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # Utility Parameters  (likelihood)
        self.sigma = nn.Parameter(torch.zeros(1))
        self.sigma_t = nn.Parameter(torch.zeros(1))

    def set_opt_max(self, is_max):
        self.opt_max = is_max

    def basis_matrix(self):
        x = torch.arange(1, self.t + 1)  # Sample x values
        self.k_values = torch.arange(0, self.n_knots) * (self.t / self.n_knots)  # Values of k
        self.rw_interval = torch.diff(self.k_values)[0].item()
        X = self.create_basis(x, self.k_values)
        return X

    def fe_basis_matrix(self):
        ''' Alternate linear basis with 'hat' type basis'''
        x = np.arange(1, self.t + 1)
        y = dmatrix(f"bs(x, df={self.n_knots}, degree=1, include_intercept=False) - 1", {"x": x})
        self.k_values = torch.arange(0, self.n_knots) * (self.t / self.n_knots)
        self.rw_interval = torch.diff(self.k_values)[0].item()
        return torch.tensor(np.array(y), dtype=torch.float32)

    def piecewise_linear(self, x, k):
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


    def create_shift_matrix(self, p, n_max, tau=10, clamp_max=1000, pad=False):
        pp = p + torch.arange(p.shape[0]).unsqueeze(1)
        POS = torch.tensor(np.arange(p.shape[0])).repeat(p.shape[0],1)
        # Filter the empty rows
        POS = POS[::self.sr,::]
        pp = pp[::self.sr]
        # Iteratively multiply the matrix by itself, clamping each loop.
        power_ = (torch.abs(POS - pp) + 0.5)
        for i in range(tau):
          power_ = power_**2
        #  # Iteratively multiply
          power_ = torch.clamp(power_, max=clamp_max)
        
        P = torch.exp(-1*(power_))

        return P

    def shift_rows_differentiable(self, x, p):
        ''' x input should be the standard input x interleaved with nan values '''
        p_expand = torch.repeat_interleave(p, self.sr)
        # Create the shift matrix form
        P = self.create_shift_matrix(p_expand, n_max=p_expand.shape[0], tau=15, clamp_max=torch.tensor(1000), pad=False)
        # Replace zeros with NANs
        #P = P[::self.sr,:] 
        # Expand the input x into the 2d shape
        B = x.view(-1, 1).expand(-1, p_expand.shape[0])

        # element wise multiply
        X_shift = P * B

        return X_shift,P

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

    def fill_nan_with_last_value(self, x):
        mask = torch.isnan(x)

        last = x[0].item()
        for i in range(1, x.shape[0]):
            if mask[i].item():
                x[i] = last
            elif mask[i].item() is False:
                last = x[i].item()
        return x

    def conditional_posterior(self, d, u, sd, n_steps):
        ''' Posterior Conditional conditional on f(X | Xn = d) Distribution
        Multivariate gaussian distribution'''
        u = (d / n_steps)
        sd = torch.sqrt((sd ** 2) * (n_steps - 1) / n_steps)
        return u, sd

    def px_z(self, u1, u2, sd1, sd2):
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

    def reverse_softplus(self, x):
        if x > -10:
          return -torch.log(1 + torch.exp(-x))
        else:
          return x

    def expected_likelihood(self, d, u, sd, n_steps):
        ''' Analytical Solution to the Expected Likelihood
        Given a random walk what is the expected likelihood
        of the walk, given that it took n_steps and moved a vertical distance d'''
        post_u, post_sd = self.conditional_posterior(d, u, sd, n_steps)
        el = torch.log(self.px_z(u1=0, u2=post_u, sd1=sd, sd2=post_sd))
        # Apply a reversed softplus so that the values are bounded to P=1
        el = self.reverse_softplus(el * n_steps)
        return el

    def terror_likelihood(self, k_values, sd, RW_width, p):
            ''' Calculate the likelihood across on the time axis
            The returned value comprises the expected likelihood, calculated
            between each of the k knot values'''
            total_likelihood = 0
            k_values = torch.cat([k_values, torch.tensor(len(p)).unsqueeze(0) - 1])
            for i in range(1, len(k_values)):
              x1 = k_values[i-1]
              x2 = k_values[i]

              y1 = p[x1.to(torch.int)]
              y2 = p[x2.to(torch.int)]

              d = y2 - y1

              el = self.expected_likelihood(d, 0, sd, RW_width)
              total_likelihood = total_likelihood + el
            return total_likelihood

    def forward(self, x, basis_X):

        # shift parameters are dependent on the previous ones so they need scaling for the gradients
        # also clamp the B parameters before hand so they dont get so large.
        #B = torch.clamp(self.B, min=-30, max=30)
        p = torch.matmul(basis_X, self.B )
        p = p/self.scale_factor

        # Shift across the rows, take the mean and forward fill.
        x_shift, P = self.shift_rows_differentiable(x, p)
        P = torch.sum(P, dim=0)
        x_shift = torch.nansum(x_shift, dim=0)


        # Convert the mask zeros to NA values
        x_shift[torch.abs(P) < 1e-3] = float('nan')

        # Forward fill the NA values
        x_shift[0] = x[0]
        x_shift = self.fill_nan_with_last_value(x_shift)
        

        # Resample and reshape
        x_shift = x_shift[::self.sr]
        x_shift = x_shift[:len(p)]
        x_shift = x_shift.unsqueeze(1)

        # Apply the fc layers
        h1 = self.relu(self.fc1(x_shift))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        y = self.fc4(h3)
        return y.squeeze(1), p

    def predict_realisations(self,x,n,rw_realisation=True):
        realisations = []
        for i in range(n):
          y = self.predict(x, True).detach().numpy()
          realisations.append(y)
        return realisations

    def predict(self, x, rw_realisation=False):
        if rw_realisation is False:
          # Predict without any time shifting
          p = torch.matmul(basis_X, torch.tensor(torch.zeros(self.B.shape[0])))
        else:
          rw = np.cumsum(np.random.normal(0, np.exp(self.sigma_t.detach().numpy())*self.sr,size=x.shape[0]*self.sr), axis=0)
          p = torch.tensor(rw[::model.sr],dtype=torch.float32)

      # Shift across the rows, take the mean and forward fill.
        x_shift, P = self.shift_rows_differentiable(x, p)
        P = torch.sum(P, dim=0)
        x_shift = torch.nansum(x_shift, dim=0)


        # Convert the mask zeros to NA values
        x_shift[torch.abs(P) < 1e-3] = float('nan')

        # Forward fill the NA values
        x_shift[0] = x[0]
        x_shift = self.fill_nan_with_last_value(x_shift)
        

        # Resample and reshape
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

        # Compute the error likelihood
        dist = Normal(loc=0, scale=sigma)
        ll = dist.log_prob(y_pred-y).sum()

        #Random walk shift likelihood
        tl = self.terror_likelihood(self.k_values, sigma_t, self.rw_interval, p)

        # Maximize for Pygad genetic, minimize for pytorch optimizers
        if self.opt_max is True:
          return (ll + tl)
        else:
          return -(ll + tl)
