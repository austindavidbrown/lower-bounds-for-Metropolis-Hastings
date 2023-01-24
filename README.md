# Estimate lower bounds on the rate of convergence for Metropolis-Hastings algorithms

A Python implementation to estimate lower bounds on the geometric convergence rate for RWM Metropolis-Hastings from the pre-print https://arxiv.org/abs/2212.05955. The library uses Pytorch. Install using PIP:

```bash
pip install mhlb
```

# Example

Here is a simple example to estimate the lower bound on the convergence rate for random-walk Metropolis-Hastings for the posterior in Bayesian ridge logistic regression:

```python
import torch
from mhlb import lb_rwm

# Generate logistic regression data
d = 10
n = 100

theta_true = torch.zeros(d).normal_(0, 1)
X = torch.zeros(n, d).uniform_(-1, 1)
X[:, 0] = 1  
Y = torch.zeros(n, dtype=torch.long)
prob = torch.sigmoid(X @ theta_true)
for i in range(0, Y.size(0)):
  Y[i] = torch.bernoulli(prob[i])

# The negative log of the target density i.e. \pi \propto \exp(-f)
sigma2_prior = 10
def negative_log_target_density(theta):
  out = X @ theta
  loss = torch.sum(torch.log1p(torch.exp(out)) - Y.double() * out ) \
         + 1/(2.0 * sigma2_prior) * theta @ theta
  return loss

# Estimates a lower bound on the geometric convergence rate for RWM
lb_rwm(negative_log_target_density, # \pi \propto \exp(-f)
            dimension = d,  # dimension of the parameter
            var_rwm = 2.38**2/d) # variance in the RWM proposal
```

## Citation

Pre-print:
https://arxiv.org/abs/2212.05955

My dissertation (Chapter 3):
https://conservancy.umn.edu/handle/11299/243073

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)

## Dissertation simulations

This project also includes some simulations from Chapter 3 of my dissertation before I made the more general package. 
