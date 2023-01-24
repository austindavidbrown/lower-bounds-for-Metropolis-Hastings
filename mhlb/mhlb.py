import torch

# Gradient descent on the the function f
# x_0: initialization point
def grad_descent(loss, x_0, 
                anneal_factor = .99,
                stepsize = .5, tol = 10**(-8), max_iterations = 10**6):
  old_loss = loss(x_0)

  x = x_0
  h = stepsize
  for t in range(1, max_iterations):    
    # Compute grad
    x_ag = x.clone().requires_grad_()  
    loss_grad = torch.autograd.grad(loss(x_ag), x_ag)[0]

    # if NAN? Throw error.
    # Otherwise? Continue.
    if torch.any(torch.isnan(loss_grad)):
      raise Exception("NAN value in gradient descent.")
    else:
      # Propose new point
      x_new = x - h * loss_grad
      new_loss = loss(x_new)

      # If loss improves? Exit or update.
      # Otherwise? Reduce step size and continue.
      if (new_loss > old_loss):
        h = h * anneal_factor
      else: 
        # Stopping criterion
        if (old_loss - new_loss) < tol:
          return x_new.detach()
        else:
          # Update
          x = x_new
          old_loss = new_loss

  # Throw error if never converged
  raise Exception("Gradient descent failed to converge")

###
# Estimate a lower bound on the convergence rate 
# for the RWM algorithm
###
# f: the negative log of the target density (i.e. \pi \propto \exp(-f))
# dimension: dimension of the parameter
# var_rwm: the variance for the Gaussian RWM proposal
# mc_iterations: number of Monte Carlo iterations to use
# returns the mean and quantiles
def lb_rwm(f, dimension, var_rwm,
           mc_iterations = 1000): 
  theta_opt = grad_descent(f, torch.zeros(dimension))
 
  # estimate accept
  estimates = torch.zeros(mc_iterations)
  for i in range(1, mc_iterations):
      f_theta_opt = f(theta_opt)
      theta_new = theta_opt + var_rwm**(1/2.) * torch.zeros(theta_opt.size(0)).normal_(0, 1)
      f_theta_new = f(theta_new)
      estimates[i] = torch.min(torch.exp(f_theta_opt - f_theta_new), torch.ones(1))
  return 1 - torch.mean(estimates).item()
