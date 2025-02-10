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
from utils import plot_tradeoff,plot_iters, plot_contours, plot_contours_line
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":24,
    "font.family": "serif"
})

# Formulate constants
n = 2
N = 500
test_perc = 0.99
# k = npr.uniform(1,4,n)
# p = k + npr.uniform(2,5,n)
k = np.array([4.,5.])
p = np.array([5,6.5])
# k_tch = torch.tensor(k, requires_grad = True)?
# p_tch = torch.tensor(p, requires_grad = True)

def gen_demand_intro(N, seed):
    np.random.seed(seed)
    sig = np.array([[0.6,-0.3],[-0.3,0.1]])
    mu = np.array((1.1,1.7))
    norms = np.random.multivariate_normal(mu,sig, N)
    d_train = np.exp(norms)
    return d_train

# Generate data
# data = gen_demand_intro(N, seed=5)
data = gen_demand_intro(N, seed=18)

num_scenarios = N
num_reps = int(N/num_scenarios)
k_data = k + np.random.normal(0,0.5,(num_scenarios,n))
p_data = k_data + np.maximum(0,np.random.normal(0,0.5,(num_scenarios,n)))
p_data = np.vstack([p_data]*num_reps)
k_data = np.vstack([k_data]*num_reps)

# Formulate uncertainty set
u = lropt.UncertainParameter(n,
                        uncertainty_set=lropt.Ellipsoidal(
                                                    data=data))
# Formulate the Robust Problem
x_r = cp.Variable(n)
t = cp.Variable()
k = lropt.Parameter(2, data=k_data)
p = lropt.Parameter(2, data=p_data)
p_x = cp.Variable(n)
objective = cp.Minimize(t)
constraints = [lropt.max_of_uncertain([-p[0]*x_r[0] - p[1]*x_r[1],-p[0]*x_r[0] - p_x[1]*u[1], -p_x[0]*u[0] - p[1]*x_r[1], -p_x[0]*u[0]- p_x[1]*u[1]]) + k@x_r <= t]
constraints += [p_x == p]
constraints += [x_r >= 0]

eval_exp = k@x_r + cp.maximum(-p[0]*x_r[0] - p[1]*x_r[1],-p[0]*x_r[0] - p[1]*u[1], -p[0]*u[0] - p[1]*x_r[1], -p[0]*u[0]- p[1]*u[1]) 

prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp)
target = -0.0
test_p = 0.9
s = 8

# setup intial A, b
train, test = train_test_split(data, test_size=int(data.shape[0]*test_p), random_state=s)
init = sc.linalg.sqrtm(np.cov(train.T))
init_bval = np.mean(train, axis=0)

np.random.seed(15)
#initn = 5*np.random.rand(n,2)
# initn = np.eye(n) + 5*np.random.rand(n,2)
# init_bvaln = np.mean(train, axis=0)
initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)
# Train A and b
from lropt import Trainer
trainer = Trainer(prob)
result = trainer.train(lr=0.001, train_size = False, num_iter=50, optimizer="SGD",seed=5, init_A=initn, init_b=init_bvaln, init_lam=0.1, init_mu=0.1,
                    mu_multiplier=1.001, kappa=0., init_alpha=0., test_percentage = test_p,save_history = True, quantiles = (0.4,0.6), lr_step_size = 50, lr_gamma = 0.5, random_init = True, num_random_init = 5, parallel = False, position = False, eta=0.3, contextual = True)
df = result.df
A_fin = result.A
b_fin = result.b