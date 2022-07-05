import torch

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Gradient descent with annealing step sizes
def graddescent(X, Y,
                stepsize = .1, tol = 10**(-10), max_iterations = 10**5):
  bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

  b = torch.zeros(1)
  theta = torch.zeros(X.size(1))

  old_loss = bceloss(b + X @ theta, Y.double())

  for t in range(1, max_iterations):
    grad_loss_b = torch.ones(X.size(0)) @ (torch.sigmoid(b + X @ theta) - Y)
    grad_loss_theta = X.T @ (torch.sigmoid(b + X @ theta) - Y)

    if torch.any(torch.isnan(grad_loss_b)) or torch.any(torch.isnan(grad_loss_theta)):
      raise Exception("NAN value in gradient descent.")
    else:
      b_new = b - stepsize * grad_loss_b
      theta_new = theta - stepsize * grad_loss_theta
      new_loss = bceloss(b_new + X @ theta_new, Y.double())
      
      # New loss worse than old loss? Reduce step size and try again.
      if (new_loss > old_loss):
        stepsize = stepsize * (.99)
      else:
        # Stopping criterion
        if (old_loss - new_loss) < tol:
          return b, theta

        # Update
        b = b_new
        theta = theta_new
        old_loss = new_loss

  raise Exception("Gradient descent failed to converge.")

# Estimate the acceptance probability at the optimum
def estimate_accept(X, Y, h_rwm, mc_iterations = 1000):
  b_opt, theta_opt = graddescent(X, Y)
  n_features = X.size(1)
  n_samples = X.size(0)
  bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

  # estimate accept
  estimates = torch.zeros(mc_iterations)
  for i in range(1, mc_iterations):
      f_theta_opt = bceloss(b_opt + X @ theta_opt, Y.double())
      theta_new = theta_opt + h_rwm**(1/2.) * torch.zeros(n_features).normal_(0, 1)
      f_theta_new = bceloss(b_opt + X @ theta_new, Y.double())
      estimates[i] = torch.min(torch.exp(f_theta_opt - f_theta_new), torch.ones(1))
  return torch.mean(estimates)

###
# Run sim
###
dimensions_list = [10, 10, 10, 10]
samples_list = [100, 200, 300, 400]
n_reps = len(samples_list)
n_iid = 100
n_variances = 4

accept_estimates = torch.zeros(n_iid, n_variances, n_reps)
for j in range(0, n_iid):
  print("iid:", j + 1)
  for k in range(0, n_variances):
    for rep in range(0, n_reps):
      n_features = 10
      n_samples = samples_list[rep]

      # Use different variance choices
      if k == 0:
        h_rwm = .1
      if k == 1:
        h_rwm = 5/n_samples
      if k == 2:
        h_rwm = 1/n_samples
      if k == 3:
        h_rwm = .1/n_samples

      # Generate data
      bias_true = 1
      theta_true = torch.zeros(n_features).normal_(0, 1)
      X = torch.zeros(n_samples, n_features)
      for i in range(0, n_samples):
        X[i, :] = torch.zeros(n_features).uniform_(-2, 2)
      Y = torch.zeros(n_samples, dtype=torch.long)
      prob = torch.sigmoid(bias_true + X @ theta_true)
      for i in range(0, Y.size(0)):
        Y[i] = torch.bernoulli(prob[i])
      
      accept_estimates[j, k, rep] = estimate_accept(X, Y, h_rwm)


###
# Convergence rate lower bound plot
###
mean_lb_estimates = (1 - accept_estimates).mean(0)
std_lb_estimates = (1 - accept_estimates).std(0)

linewidth = 3
markersize = 5
alpha = .8

light_blue_color = (3./255, 37./255, 76./255)
dark_blue_color = (24./255, 123./255, 205./255)

red_color = (0.86, 0.3712, 0.33999999999999997)
blue_color = (0.33999999999999997, 0.43879999999999986, 0.86)
green_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
purple_color = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)


plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

iterations = torch.arange(0, n_reps)
samples_and_dimensions = list(zip(dimensions_list, samples_list))

def plot_estimate(mean_estimates, std_estimates, color, label):
  plt.plot(iterations, mean_estimates, 
           '-', alpha = alpha, marker="v", markersize=markersize, color=color, label=label, linewidth = linewidth)
  plt.fill_between(iterations, mean_estimates - std_estimates/n_iid**(1/2.),
                   mean_estimates + std_estimates/n_iid**(1/2.), alpha=0.1,
                   color=color)


plot_estimate(mean_lb_estimates[0, :].numpy(), 
              std_lb_estimates[0, :].numpy(), 
              red_color,
              r"$h = .1$")

plot_estimate(mean_lb_estimates[1, :].numpy(), 
              std_lb_estimates[1, :].numpy(), 
              green_color,
              r"$h = 5/n$")

plot_estimate(mean_lb_estimates[2, :].numpy(), 
              std_lb_estimates[2, :].numpy(), 
              purple_color,
              r"$h = 1/n$")

plot_estimate(mean_lb_estimates[3, :].numpy(), 
              std_lb_estimates[3, :].numpy(), 
              dark_blue_color,
              r"$h = .1/n$")

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Convergence rate lower bound", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("flat_acceptance_plot.png", pad_inches=0, bbox_inches='tight',)


####
# Mixing time lower bound plot
####
mean_mt_estimates = (1/accept_estimates - 1).mean(0)
std_mt_estimates = (1/accept_estimates - 1).std(0)

plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

iterations = torch.arange(0, n_reps)
samples_and_dimensions = list(zip(dimensions_list, samples_list))

plot_estimate(mean_mt_estimates[0, :].numpy(), 
              std_mt_estimates[0, :].numpy(), 
              red_color,
              r"$h = .1$")

plot_estimate(mean_mt_estimates[1, :].numpy(), 
              std_mt_estimates[1, :].numpy(), 
              green_color,
              r"$h = 5/n$")

plot_estimate(mean_mt_estimates[2, :].numpy(), 
              std_mt_estimates[2, :].numpy(), 
              purple_color,
              r"$h = 1/n$")


plot_estimate(mean_mt_estimates[3, :].numpy(), 
              std_mt_estimates[3, :].numpy(), 
              dark_blue_color,
              r"$h = .1/n$")

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Mixing time lower bound", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("flat_mt_plot.png", pad_inches=0, bbox_inches='tight',)






