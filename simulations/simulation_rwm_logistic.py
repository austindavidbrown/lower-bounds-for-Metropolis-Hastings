import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Pytorch settings
import torch
torch.manual_seed(5)
torch.autograd.set_grad_enabled(False)

# Compute the binary cross entropy loss for logistic regression
# Replaced torch.nn.BCEWithLogitsLoss(reduction="sum") just to be more explicit
def bceloss(out, Y):
  return torch.sum( torch.log1p(torch.exp(out)) - Y * out )

# Gradient descent on the negative log of the target density with annealing step sizes
def graddescent(X, Y, Cov_prior, 
                anneal_factor = .99,
                stepsize = .5, tol = 10**(-8), max_iterations = 10**6):
  Cov_prior_half = torch.linalg.cholesky(Cov_prior)
  Cov_prior_inv = torch.cholesky_inverse(Cov_prior_half)

  b = torch.zeros(1)
  theta = torch.zeros(X.size(1))

  old_loss = bceloss(b + X @ theta, Y.double()) \
             + 1/(2.0) * theta @ Cov_prior_inv @ theta

  for t in range(1, max_iterations):
    grad_loss_b = torch.ones(X.size(0)) @ (torch.sigmoid(b + X @ theta) - Y)
    grad_loss_theta = X.T @ (torch.sigmoid(b + X @ theta) - Y) + Cov_prior_inv @ theta

    if torch.any(torch.isnan(grad_loss_b)) or torch.any(torch.isnan(grad_loss_theta)):
      raise Exception("NAN value in gradient descent.")
    else:
      b_new = b - stepsize * grad_loss_b
      theta_new = theta - stepsize * grad_loss_theta
      new_loss = bceloss(b_new + X @ theta_new, Y.double()) \
                 + 1/(2.0) * theta_new @ Cov_prior_inv @ theta_new

      # New loss worse than old loss? Reduce step size and try again.
      if (new_loss > old_loss):
        stepsize = stepsize * anneal_factor
      else:
        # Stopping criterion
        if (old_loss - new_loss) < tol:
          return b_new, theta_new

        # Update
        b = b_new
        theta = theta_new
        old_loss = new_loss

  raise Exception("Gradient descent failed to converge.")

# Estimate the acceptance probability for RWM at the optimum of the target density
# Estimate the bias with the MLE (i.e. don't use MCMC for the bias)
def estimate_accept(X, Y, Cov_prior, h_rwm,
                    b_opt, theta_opt,
                    mc_iterations = 1000):
  n_features = X.size(1)
  n_samples = X.size(0)

  Cov_prior_half = torch.linalg.cholesky(Cov_prior)
  Cov_prior_inv = torch.cholesky_inverse(Cov_prior_half)

  # estimate accept
  estimates = torch.zeros(mc_iterations)
  for i in range(1, mc_iterations):
      f_theta_opt = bceloss(b_opt + X @ theta_opt, Y.double()) \
                    + 1/(2.0) * theta_opt @ Cov_prior_inv @ theta_opt
      theta_new = theta_opt + h_rwm**(1/2.) * torch.zeros(n_features).normal_(0, 1)
      f_theta_new = bceloss(b_opt + X @ theta_new, Y.double()) \
                    + 1/(2.0) * theta_new @ Cov_prior_inv @ theta_new
      estimates[i] = torch.min(torch.exp(f_theta_opt - f_theta_new), torch.ones(1))
  return torch.mean(estimates)

###
# Run simulation
###
gamma = 4
g = 10

dimensions_list = [2, 4, 6, 8, 10, 12, 14]
samples_list = [int(gamma * k) for k in dimensions_list]
n_reps = len(dimensions_list)
n_iid = 50
n_variances = 3

accept_estimates = torch.zeros(n_iid, n_variances, n_reps)
for j in range(0, n_iid):
  print("iid:", j + 1)
  for rep in range(0, n_reps):
    n_features = dimensions_list[rep]
    n_samples = samples_list[rep]

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

    # Create the prior covariance
    XTX_half = torch.linalg.cholesky(X.T @ X)
    XTX_inv = torch.cholesky_inverse(XTX_half)
    Cov_prior = g * XTX_inv

    b_opt, theta_opt = graddescent(X, Y, Cov_prior)

    for k in range(0, n_variances):
      # Use different variance choices
      if k == 0:
        h_rwm = .6
      if k == 1:
        h_rwm = 2.38**2/n_features
      if k == 2:
        h_rwm = 1/(n_samples * n_features)
      
      accept_estimates[j, k, rep] = estimate_accept(X, Y, Cov_prior, h_rwm,
                                                    b_opt, theta_opt)


###
# Plotting settings
###
linewidth = 3
markersize = 5
alpha = .8

light_blue_color = (3./255, 37./255, 76./255)
dark_blue_color = (24./255, 123./255, 205./255)

red_color = (0.86, 0.3712, 0.33999999999999997)
blue_color = (0.33999999999999997, 0.43879999999999986, 0.86)
green_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
purple_color = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)

solid_line = (0, ()) 
dense_dashed_line = (0, (5, 1))
dash_dot_line = (0, (3, 1, 1, 1))

###
# Convergence rate lower bound plot
###
mean_lb_estimates = (1 - accept_estimates).mean(0)
std_lb_estimates = (1 - accept_estimates).std(0)

plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

iterations = torch.arange(0, n_reps)
samples_and_dimensions = list(zip(dimensions_list, samples_list))


plt.plot(iterations, mean_lb_estimates[0, :].cpu().numpy(), 
         linestyle=solid_line, alpha = alpha, marker="v", markersize=markersize, color=dark_blue_color, label=r"$h = .6$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[0, :] - std_lb_estimates[0, :]/n_iid**(1/2.),
                 mean_lb_estimates[0, :] + std_lb_estimates[0, :]/n_iid**(1/2.), alpha=0.1,
                 color=dark_blue_color)

plt.plot(iterations, mean_lb_estimates[1, :].cpu().numpy(), 
         linestyle=dense_dashed_line, alpha = alpha, marker="v", markersize=markersize, color=green_color, label=r"$h = 2.38^2/d$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[1, :] - std_lb_estimates[1, :]/n_iid**(1/2.),
                 mean_lb_estimates[1, :] + std_lb_estimates[1, :]/n_iid**(1/2.), alpha=0.1,
                 color=green_color)

plt.plot(iterations, mean_lb_estimates[2, :].cpu().numpy(), 
         linestyle=dash_dot_line, alpha = alpha, marker="v", markersize=markersize, color=purple_color, label=r"$h = 1/(d n)$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[2, :] - std_lb_estimates[2, :]/n_iid**(1/2.),
                 mean_lb_estimates[2, :] + std_lb_estimates[2, :]/n_iid**(1/2.), alpha=0.1,
                 color=purple_color)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Convergence rate lower bound", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("acceptance_plot.png", pad_inches=0, bbox_inches='tight',)



###
# log acceptance probability
###
mean_lb_estimates = torch.log(accept_estimates).mean(0)
std_lb_estimates = torch.log(accept_estimates).std(0)

plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

iterations = torch.arange(0, n_reps)
samples_and_dimensions = list(zip(dimensions_list, samples_list))


plt.plot(iterations, mean_lb_estimates[0, :].cpu().numpy(), 
         linestyle=solid_line, alpha = alpha, marker="v", markersize=markersize, color=dark_blue_color, label=r"$h = .6$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[0, :] - std_lb_estimates[0, :]/n_iid**(1/2.),
                 mean_lb_estimates[0, :] + std_lb_estimates[0, :]/n_iid**(1/2.), alpha=0.1,
                 color=dark_blue_color)

plt.plot(iterations, mean_lb_estimates[1, :].cpu().numpy(), 
         linestyle=dense_dashed_line, alpha = alpha, marker="v", markersize=markersize, color=green_color, label=r"$h = 2.38^2/d$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[1, :] - std_lb_estimates[1, :]/n_iid**(1/2.),
                 mean_lb_estimates[1, :] + std_lb_estimates[1, :]/n_iid**(1/2.), alpha=0.1,
                 color=green_color)

plt.plot(iterations, mean_lb_estimates[2, :].cpu().numpy(), 
         linestyle=dash_dot_line, alpha = alpha, marker="v", markersize=markersize, color=purple_color, label=r"$h = 1/(d n)$", linewidth = linewidth)
plt.fill_between(iterations, mean_lb_estimates[2, :] - std_lb_estimates[2, :]/n_iid**(1/2.),
                 mean_lb_estimates[2, :] + std_lb_estimates[2, :]/n_iid**(1/2.), alpha=0.1,
                 color=purple_color)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Log acceptance probability", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("log_acceptance_plot.png", pad_inches=0, bbox_inches='tight',)


