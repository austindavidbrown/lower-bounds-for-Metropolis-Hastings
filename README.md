# Lower bounds on the rate of convergence for Metropolis-Hastings algorithms

A Python implementation to estimate lower bounds on the geometric convergence rate for RWM Metropolis-Hastings. This library relies on Pytorch. Install using PIP:

```bash
pip install mhlb
```

# Example

Here is a simple example:

```python
import torch
from mhlb import estimate_lower_bound_RWM

# Generate logistic regression data
n_features = 10
n_samples = 100

theta_true = torch.zeros(n_features + 1).normal_(0, 1)
X = torch.zeros(n_samples, n_features).uniform_(-1, 1)  
Y = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(theta_true[0] + X @ theta_true[1:])
for i in range(0, Y.size(0)):
  Y[i] = torch.bernoulli(prob[i])

# Create a target distribution class
class RidgeLogisticRegressionPosterior:
  def __init__(self, X, Y, sigma2_prior):
    self.X = X
    self.Y = Y
    self.g = g

  # \pi \propto \exp(-f)
  def f(self, theta):
    out = theta[0] + self.X @ theta[1:]
    loss = torch.sum(torch.log1p(torch.exp(out)) - self.Y.double() * out ) \
           + 1/(2.0 * sigma2_prior) * theta[1:] @ theta[1:]
    return loss

target_distribution = RidgeLogisticRegressionPosterior(X, Y, sigma2_prior = 10)

# Estimate a lower bound on the geometric convergence rate
lb = lb_rwm(f = target_distribution.f, 
            dimension = n_features + 1, 
            var_rwm = 2.38**2/dimension)
print(lb)
```

## Citation

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)
* [Seaborn](https://seaborn.pydata.org)
* [Matplotlib]([https://seaborn.pydata.org](https://matplotlib.org))
