import argparse
import os
import sys
import joblib
from joblib import Parallel, delayed
output_stream = sys.stdout
import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import lropt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import warnings
warnings.filterwarnings("ignore")

test_p = 0.5
N = 2000
n = 10
m = 8
np.random.seed(27)
y_nom = np.random.uniform(2,4,n)
y_data = y_nom
num_context = 20
num_reps = int(N/num_context)
for scene in range(num_context-1):
    np.random.seed(scene)
    y_data = np.vstack([y_data,np.maximum(y_nom + np.random.normal(0,0.05,n),0)])
np.random.seed(27)
C = 200
c = np.random.uniform(30,50,n)
Q = np.random.uniform(-0.2,0.2,(n,m))
d = np.random.uniform(10,20,n)
t = np.random.uniform(0.1,0.3,n)
h = np.random.uniform(0.1,0.3,n)

def gen_sigmu_varied(n,m,N = 500,seed = 0):
    np.random.seed(seed)
    sig = []
    context = []
    for i in range(N):
        F = np.random.normal(size = (n,m))
        context.append(F)
        csig = 0.2*F@(F.T)
        sig.append(csig)
    return np.stack(sig), np.stack(context)

def gen_demand_varied(sig,mu,d,N,seed=399):
    pointlist = []
    np.random.seed(seed)
    for i in range(N):
        d_train = np.random.multivariate_normal(d - 0.1*mu[i],sig[i])
        pointlist.append(d_train)
    return np.vstack(pointlist)

sig, context = gen_sigmu_varied(n,m,num_context,seed= 0)
sig = np.vstack([sig]*num_reps)
context_dat = np.vstack([context]*num_reps)
y_data = np.vstack([y_data]*num_reps)

test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
test_indices = test_valid_indices[:int((test_p)*N)]
valid_indices = test_valid_indices[int((test_p)*N):]
train_indices = [i for i in range(N) if i not in test_valid_indices]
context_inds = {}
test_inds = {}
for j in range(num_context):
  context_inds[j]= [i for i in train_indices if j*num_reps <= i <= (j+1)*num_reps]
  test_inds[j] = [i for i in test_valid_indices if j*num_reps <= i <= (j+1)*num_reps]

seed = 0
data = gen_demand_varied(sig,y_data,d,N,seed= 0)
train = data[train_indices]
init_bval = np.mean(train, axis=0)
init = np.real(sc.linalg.sqrtm(np.cov(train.T)))

u = lropt.UncertainParameter(n,
                                uncertainty_set = lropt.Ellipsoidal(p=2, data =data))
# formulate cvxpy variable
L = cp.Variable()
s = cp.Variable(n)
y = cp.Variable(n)
Y = cp.Variable((n,n))
r = lropt.ContextParameter(n, data = y_data)
context = lropt.ContextParameter((n,m), data=context_dat)     
Y_r = cp.Variable(n)
# formulate objective
objective = cp.Minimize(L)

# formulate constraints
constraints = []
constraints += [context >= -200]
cons = [-r@y - Y_r@u + (t+h)@s - L]
for idx in range(n):
    cons += [y[idx]+Y[idx]@u-s[idx]]
    cons += [y[idx]+Y[idx]@u-u[idx]]
constraints += [lropt.max_of_uncertain(cons)<=0]
constraints += [r@Y == Y_r]
constraints += [np.ones(n)@s == C]
constraints += [s <=c, s >=0]
eval_exp = -r@y - r@Y@u + (t+h)@s
# formulate Robust Problem
prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp )
# solve
# seed 1, 
trainer = lropt.Trainer(prob)
settings = lropt.TrainerSettings()
settings.set(lr = 0.001,num_iter=100, optimizer = "SGD", seed = 5, init_A = init, init_b = init_bval, init_lam = 2.0, init_mu =2.0, mu_multiplier=1.02, init_alpha = -0.0, test_percentage = test_p, save_history = False, lr_step_size = 50, lr_gamma = 0.5, position = False, random_init = False, num_random_init=6, parallel = False, eta = 0.3, kappa=0.,contextual = True,validate_frequency = 20,test_frequency = 300,predictor = lropt.LinearPredictor(predict_mean = True,pretrain=True, lr=0.001,epochs = 100))
result = trainer.train(settings = settings)
A_fin = result.A
b_fin = result.b