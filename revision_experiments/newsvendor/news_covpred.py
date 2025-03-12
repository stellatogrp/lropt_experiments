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
k_init = np.array([4.,5.])
p = np.array([5,6.5])
# k_tch = torch.tensor(k, requires_grad = True)
# p_tch = torch.tensor(p, requires_grad = True)

def gen_demand_intro(N, seed):
    np.random.seed(seed)
    sig = np.array([[0.6,-0.3],[-0.3,0.1]])
    mu = np.array((1.1,1.7))
    norms = np.random.multivariate_normal(mu,sig, N)
    d_train = np.exp(norms)
    return d_train

def gen_demand_cor(N,seed,x):
    np.random.seed(seed)
    sig = np.eye(2)
    mu = np.array((6,7))
    points_list = []
    for i in range(N):
        mu_shift = -0.4*x[i]
        newpoint = np.random.multivariate_normal(mu+mu_shift,sig)
        points_list.append(newpoint)
    return np.vstack(points_list)

test_p = 0.9
s = 8
np.random.seed(s)
num_scenarios = N
num_reps = int(N/num_scenarios)
k_data = np.maximum(0.5,k_init + np.random.normal(0,3,(num_scenarios,n)))
p_data = k_data + np.maximum(0,np.random.normal(0,3,(num_scenarios,n)))
p_data = np.vstack([p_data]*num_reps)
k_data = np.vstack([k_data]*num_reps)

data = gen_demand_cor(N,seed=5,x=p_data)

# setup intial A, b
test_indices = np.random.choice(N,int(0.9*N), replace=False)
train_indices = [i for i in range(N) if i not in test_indices]
train = np.array([data[i] for i in train_indices])
test = np.array([data[i] for i in test_indices])
k_train = np.array([k_data[i] for i in train_indices])
k_test = np.array([k_data[i] for i in test_indices])
p_train = np.array([p_data[i] for i in train_indices])
p_test = np.array([p_data[i] for i in test_indices])

# Formulate uncertainty set
u = lropt.UncertainParameter(n,
                        uncertainty_set=lropt.Ellipsoidal(
                                                    data=data))
# Formulate the Robust Problem
x_r = cp.Variable(n)
t = cp.Variable()
k = lropt.ContextParameter(2, data=k_data)
p = lropt.ContextParameter(2, data=p_data)
p_x = cp.Variable(n)
objective = cp.Minimize(t)
constraints = [lropt.max_of_uncertain([-p[0]*x_r[0] - p[1]*x_r[1],-p[0]*x_r[0] - p_x[1]*u[1], -p_x[0]*u[0] - p[1]*x_r[1], -p_x[0]*u[0]- p_x[1]*u[1]]) + k@x_r <= t]
constraints += [p_x == p]
constraints += [x_r >= 0]

eval_exp = k@x_r + cp.maximum(-p[0]*x_r[0] - p[1]*x_r[1],-p[0]*x_r[0] - p[1]*u[1], -p[0]*u[0] - p[1]*x_r[1], -p[0]*u[0]- p[1]*u[1]) 

prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp)
target = -0.0

# setup intial A, b
init = sc.linalg.sqrtm(np.cov(train.T))
init_bval = np.mean(train, axis=0)

np.random.seed(15)
#initn = 5*np.random.rand(n,2)
# initn = np.eye(n) + 5*np.random.rand(n,2)
# init_bvaln = np.mean(train, axis=0)
initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)
# init_bias = np.hstack([initn.flatten(),np.array([6,7])])
# init_weight = np.vstack([np.zeros((4,4)),np.array([[-0.4,0,0,0],[0,-0.4,0,0]])])
# init_bias = np.hstack([initn.flatten(),mults_mean_bias])
# init_weight = np.vstack([np.zeros((4,4)),mults_mean_weight])

# Train A and b
from lropt import Trainer
trainer = Trainer(prob)
trainer_settings = lropt.TrainerSettings()
trainer_settings.lr=0.0001
trainer_settings.train_size = False
trainer_settings.num_iter=1
trainer_settings.optimizer="SGD"
trainer_settings.seed=5
trainer_settings.init_A=initn
trainer_settings.init_b=init_bvaln
trainer_settings.init_lam=0.5
trainer_settings.init_mu=0.5
trainer_settings.mu_multiplier=1.001
trainer_settings.test_percentage = test_p
trainer_settings.save_history = True
trainer_settings.quantiles = (0.4,0.6)
trainer_settings.lr_step_size = 50
trainer_settings.lr_gamma = 0.5
trainer_settings.random_init = False
trainer_settings.num_random_init = 3
trainer_settings.parallel = False
trainer_settings.position = False
trainer_settings.eta=0.3
trainer_settings.contextual = True
trainer_settings.covpred = True
result = trainer.train(trainer_settings=trainer_settings)
df = result.df
A_fin = result.A
b_fin = result.b
eps_list = np.linspace(0.5, 2.5, 10)
result5 = trainer.grid(rholst=eps_list,init_A=A_fin, init_b=b_fin, seed=s,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, covpred = True)
dfgrid2 = result5.df