import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import lropt
import sys
sys.path.append('..')
from utils import plot_iters
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":24,
    "font.family": "serif"
})

import scipy.sparse as spa
from cvxpylayers.torch import CvxpyLayer
import networkx as nx
from lropt import Trainer

np.random.seed(0)

# generate problem data
n = 4   # nodes
k = 2   # suppliers (with prices p)
c = 2   # retail (with demand d)
m = 8   # links

supply_links = [0, 1]
retail_links = [6, 7]
internode_links = [2, 3, 4, 5]

# Incidence matrices (nodes x links)
A_in = np.array([[1, 0, 0, 0, 0, 0, 0, 0],   # 1 (supply)
                 [0, 1, 0, 0, 0, 0, 0, 0],   # 2 (supply)
                 [0, 0, 1, 0, 0, 1, 0, 0],   # 3 (retail)
                 [0, 0, 0, 1, 1, 0, 0, 0],   # 4 (retail)
                 ])

A_out = np.array([[0, 0, 1, 1, 0, 0, 0, 0],   # 1 (supply)
                  [0, 0, 0, 0, 1, 0, 0, 0],   # 2 (supply)
                  [0, 0, 0, 0, 0, 0, 1, 0],   # 3 (retail)
                  [0, 0, 0, 0, 0, 1, 0, 1],   # 4 (retail)
                  ])

# Prices
mu_p = torch.tensor([0, 0.1]).double()
sigma_p = torch.tensor([0.2, 0.2]).double()
mean_p = torch.exp(mu_p + sigma_p ** 2 /2).double().view(k, 1)
var_p = (torch.exp(sigma_p ** 2) - 1) * torch.exp(2 * mean_p + sigma_p ** 2)

# Demands
mu_d = torch.tensor([0.0, 0.4]).double()
sigma_d = torch.tensor([0.2, 0.2]).double()
mean_d = torch.exp(mu_d + sigma_d ** 2 /2).double().view(c, 1)
var_d = (torch.exp(sigma_d ** 2) - 1) * torch.exp(2 * mean_d + sigma_d ** 2)

# Uncertainty distribution (prices and demands)
w_dist = torch.distributions.log_normal.LogNormal(torch.cat([mu_p, mu_d], 0), 
                                                  torch.cat([sigma_p, sigma_d], 0))

# Capacities
h_max = 3. # Maximum capacity in every node
u_max = 2. # Link flow capacity

# Storage cost parameters, W(x) = alpha'x + beta'x^2 + gamma
alpha = 0.01
beta = 0.01

# Transportation cost parameters
tau = 0.05 * np.ones((m - k - c,1))
tau_th = torch.tensor(tau, dtype=torch.double)
r = 1.3 * np.ones((k,1))
r_th = torch.tensor(r, dtype=torch.double)

init_size = 100
# Define linear dynamics
# x = (h, p^{wh}, d) 
# u = u
# w = (p^{wh}, d)
# x_{t+1} = Ax_{t} + Bu_{t} + w
A_d = np.bmat([[np.eye(n), np.zeros((n, k+c))],
              [np.zeros((k+c, n)), np.zeros((k+c, k+c))]])
A_d_th = torch.tensor(A_d, dtype=torch.double)
B_d = np.vstack([A_in - A_out,
                 np.zeros((k+c, m))])
B_d_th = torch.tensor(B_d, dtype=torch.double)
n_x, n_u = B_d.shape

# Setup policy
# Parameters
P_sqrt = cp.Parameter((n, n))
q = cp.Parameter((n, 1))
x = lropt.Parameter((n_x,1), data=np.zeros((init_size,n_x,1)))
# h = lropt.Parameter(n,data=np.zeros((init_size,n)))
# p = lropt.Parameter(k,data=np.zeros((init_size,k)))
d = lropt.UncertainParameter(c,uncertainty_set = lropt.Ellipsoidal(rho=1,data=np.zeros((init_size,c))))
h, p, d_unneeded = x[:n], x[n:n+k], x[(n+k):]

# Variables
u = cp.Variable((n_u,1))
h_next = cp.Variable((n,1))

# Cvxpy Layer
stage_cost = cp.vstack([p, tau, -r]).T @ u + cp.sum_squares(h_next) -3*np.ones(n)@h_next

# next_stage_cost = cp.sum_squares(P_sqrt @ h_next) + q.T @ h_next
constraints = [h_next == h + (A_in - A_out) @ u, 
               h_next <= h_max,  
               0 <= u, u <= u_max,
               A_out @ u <= h, u[retail_links].flatten()  <= d,
              ]
prob = lropt.RobustProblem(cp.Minimize(stage_cost), constraints)
trainer = Trainer(prob)
rho_mult_tch = trainer.gen_rho_mult_tch(trainer._rho_mult_parameter)
policy = trainer.create_cvxpylayer(variables = [u])

class SupplyChain(lropt.train.trainer.Simulator):
  def simulate(self,x,u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    A_batch = A_d_th.repeat(batch_size, 1, 1)
    B_batch = B_d_th.repeat(batch_size, 1, 1)
    
    zer = torch.zeros(batch_size, n, 1).double()
    w = w_dist.sample((batch_size,)).double().view((batch_size, k + c, 1))
    w_batch = torch.cat([zer, w], 1).double()
    
    return torch.bmm(A_batch, x) + torch.bmm(B_batch, u) + w_batch

  def stage_cost(self,x,u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    r_batch = r_th.repeat(batch_size, 1, 1)
    tau_batch = tau_th.repeat(batch_size, 1, 1)
    h, p, dh = x[:,:n], x[:, n:n+k], x[:, n+k:]
    s_vec = torch.cat([p, tau_batch, -r_batch], 1).double()
    S = torch.bmm(s_vec.transpose(1, 2), u)
    H = alpha * h + beta * (h ** 2)
    return torch.sum(S, 1) + torch.sum(H, 1)

  def constraint_cost(self,x,u,alpha):
    eta = 0.05
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    h, p, dh = x[:,:n], x[:, n:n+k], x[:, n+k:]
    cvar_term =(1/eta)*(torch.max(torch.max(u[:,retail_links,:] - dh,axis=1)[0] - alpha,torch.zeros(batch_size))[0]) + alpha
    return 0.01*cvar_term

  def init_state(self,batch_size, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x_b_0 = h_max * torch.rand(batch_size, n, 1).double()
    w_0 = w_dist.sample((batch_size,)).double().view((batch_size, k+c, 1))
    x_batch = torch.cat([x_b_0, w_0], 1).double()
    return x_batch
  
simulator = SupplyChain()

# Perform training
time_horizon = 20
epochs = 50
batch_size = 10
trials = 10
lr = 0.8
init_x = simulator.init_state(seed = 0, batch_size = 100)
_,_, init_dh = init_x[:,:n], init_x[:, n:n+k], init_x[:, n+k:]
init_dh = torch.flatten(init_dh, start_dim = 1)
init_a = torch.tensor(sc.linalg.sqrtm(np.cov(init_dh.detach().numpy().T)),dtype=torch.double)
init_b = torch.mean(init_dh,axis=0)

val_costs, val_costs_constr, \
  paramvals, x_base, u_base = trainer.multistage_train(simulator, 
                                                       policy = policy, 
                         time_horizon = time_horizon, epochs = epochs, 
                         batch_size = batch_size,
                         trials = trials, init_eps=1, seed=0,
                          init_a = init_a, init_b = init_b,
                          optimizer = "SGD",lr= lr, momentum = 0, scheduler = False)