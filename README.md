# Estimate lower bounds on the rate of convergence for Metropolis-Hastings algorithms
(IMPORTANT: NOT COMPLETED AND UPLOADED TO PIP YET)

A Python implementation to estimate lower bounds on the geometric convergence rate for RWM Metropolis-Hastings. See my dissertation (Chapter 3):
https://conservancy.umn.edu/handle/11299/243073. This package relies on Pytorch. 

Install using PIP:

```bash
pip install mhlb
```

This also includes some simulations from my dissertation before I made the more general package. 

# Example

Here is a simple example:

```python
import torch
from mhlb import lb_rwm

# Generate logistic regression data
n_features = 10
n_samples = 100
sigma2_prior = 10

theta_true = torch.zeros(n_features + 1).normal_(0, 1)
X = torch.zeros(n_samples, n_features).uniform_(-1, 1)  
Y = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(theta_true[0] + X @ theta_true[1:])
for i in range(0, Y.size(0)):
  Y[i] = torch.bernoulli(prob[i])

# The negative log of the target density i.e. \pi \propto \exp(-f)
def negative_log_target_density(theta):
  out = theta[0] + X @ theta[1:]
  loss = torch.sum(torch.log1p(torch.exp(out)) - Y.double() * out ) \
         + 1/(2.0 * sigma2_prior) * theta[1:] @ theta[1:]
  return loss

# Estimate a lower bound on the geometric convergence rate for RWM
dimension = n_features + 1 
lb_rwm(negative_log_target_density, # \pi \propto \exp(-f)
            dimension = dimension,  # dimension of the parameter
            var_rwm = 2.38**2/dimension) # variance in the RWM proposal
```

## Citation

My dissertation (Chapter 3):
https://conservancy.umn.edu/handle/11299/243073

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)
* [Seaborn](https://seaborn.pydata.org)
* [Matplotlib]([https://seaborn.pydata.org](https://matplotlib.org))
