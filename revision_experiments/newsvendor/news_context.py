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
k_init = np.array([4.,5.])

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
        

s = 1
np.random.seed(s)
num_reps = int(N/N)
k_data = np.maximum(0.5,k_init + np.random.normal(0,3,(N,n)))
p_data = k_data + np.maximum(0,np.random.normal(0,3,(N,n)))
p_data = np.vstack([p_data]*num_reps)
k_data = np.vstack([k_data]*num_reps)

data = gen_demand_cor(N,seed=5,x=p_data)
test_p = 0.9

# split dataset
test_indices = np.random.choice(N,int(0.9*N), replace=False)
train_indices = [i for i in range(N) if i not in test_indices]
train = np.array([data[i] for i in train_indices])
test = np.array([data[i] for i in test_indices])
k_train = np.array([k_data[i] for i in train_indices])
k_test = np.array([k_data[i] for i in test_indices])
p_train = np.array([p_data[i] for i in train_indices])
p_test = np.array([p_data[i] for i in test_indices])

def gen_weights_bias(k_data,p_data, data):
    stacked_context = np.hstack([p_data,k_data,np.ones((N,1))])
    mults_1 = np.linalg.lstsq(stacked_context,data[:,0])[0]
    mults_2 = np.linalg.lstsq(stacked_context,data[:,1])[0]
    mults_mean = np.vstack([mults_1, mults_2])
    mults_mean_weight = mults_mean[:,:-1]
    mults_mean_bias = mults_mean[:,-1]
    return mults_mean_weight, mults_mean_bias
mults_mean_weight, mults_mean_bias = gen_weights_bias(k_data,p_data, data)

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

initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)

# initialize linear weights and bias
init_bias = np.hstack([initn.flatten(),mults_mean_bias])
init_weight = np.vstack([np.zeros((4,4)),mults_mean_weight])

# Train A and b
from lropt import Trainer
trainer = Trainer(prob)
trainer_settings = lropt.TrainerSettings()

trainer_settings.lr=0.0001
trainer_settings.num_iter=300
trainer_settings.optimizer="SGD"
trainer_settings.seed=5
trainer_settings.init_A=initn
trainer_settings.init_b=init_bvaln
trainer_settings.init_lam=0.5
trainer_settings.init_mu=0.5
trainer_settings.mu_multiplier=1.001
trainer_settings.test_percentage = test_p
trainer_settings.save_history = True
trainer_settings.lr_step_size = 50
trainer_settings.lr_gamma = 0.5
trainer_settings.random_init = False
trainer_settings.num_random_init = 1
trainer_settings.parallel = True
trainer_settings.position = False
trainer_settings.eta=0.3
trainer_settings.contextual = True
trainer_settings.init_weight = init_weight
trainer_settings.init_bias = init_bias
result = trainer.train(trainer_settings=trainer_settings)
df = result.df
A_fin = result.A
b_fin = result.b

