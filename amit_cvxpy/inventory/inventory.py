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
import warnings
import sys
# import ot
sys.path.append('..')
from utils import plot_iters, plot_coverage_all
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":18,
    "font.family": "serif"
})
colors = ["tab:blue", "tab:green", "tab:orange", 
          "blue", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "red"]

def data_scaled(N, m, scale, seed):
    np.random.seed(seed)
    R = np.vstack([np.random.normal(
        i*0.03*scale, np.sqrt((0.02**2+(i*0.1)**2)), N) for i in range(1, m+1)])
    return (R.transpose())

def data_modes(N, m, scales, seed):
    modes = len(scales)
    d = np.zeros((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,
          :] = data_scaled(weights, m, scales[i], seed)
    return d[0:N, :]

n = 10
m = 4

np.random.seed(27)
C = 200
c = np.random.uniform(30,50,n)
Q = np.random.uniform(-0.2,0.2,(n,m))
d = np.random.uniform(10,20,n)
t = np.random.uniform(0.1,0.3,n)
h = np.random.uniform(0.1,0.3,n)

# saved_s = np.load("scenarios.npy")
# num_scenarios = 5
# y_data = saved_s[0]
# for scene in range(1,num_scenarios):
#     y_data = np.vstack([y_data,saved_s[scene]])

np.random.seed(5)
r = np.random.uniform(2,4,n)
y_data = r
num_scenarios = 5
for scene in range(num_scenarios):
    np.random.seed(scene)
    y_data = np.vstack([y_data,np.maximum(r + np.random.normal(0,0.1,n),0)])

seed = 27
n = 10
m = 4
N=600
test_p = 0.2
data = data_modes(N,m,[1,2,3],seed = seed)
train, test = train_test_split(data, test_size=int(
data.shape[0]*test_p), random_state=seed)
init = np.real(sc.linalg.sqrtm(sc.linalg.inv(np.diag(np.ones(m)*0.0001)+ np.cov(train.T))))
newdata = data_modes(8001,m,[1,2,3],seed = 10000+seed)
init_bval = -init@np.mean(train, axis=0)

# formulate the ellipsoidal set
u = lropt.UncertainParameter(m,
                                uncertainty_set = lropt.Ellipsoidal(p=2, data =data))
# formulate cvxpy variable
L = cp.Variable()
s = cp.Variable(n)
y = cp.Variable(n)
Y = cp.Variable((n,m))
r = lropt.Parameter(n, data = y_data)        

# formulate objective
objective = cp.Minimize(L)

# formulate constraints
constraints = [-r@y - r@Y@u + (t+h)@s <= L]
for i in range(n):
    constraints += [y[i]+Y[i]@u <= s[i]]
    constraints += [y[i]<= d[i]+ (Q[i] - Y[i])@u]
constraints += [np.ones(n)@s == C]
constraints += [s <=c, s >=0]
eval_exp = -r@y - r@Y@u + (t+h)@s
# formulate Robust Problem
prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp )

# solve
# seed 1, 
warnings.filterwarnings("ignore")
result = prob.train(lr = 0.001,num_iter=5, optimizer = "SGD", seed = seed, init_A = 0.8*init, init_b = 0.8*init_bval, init_lam = 1, init_mu = 1.5, mu_multiplier=1.01, init_alpha = -0.01, test_percentage = test_p, save_history = False, lr_step_size = 100, lr_gamma = 0.2, position = False, random_init = True, num_random_init=5, parallel = True, kappa=0, eta=0.05)
A_fin = result.A
b_fin = result.b

result5 = prob.grid(epslst = np.linspace(0.01,5, 8), init_A = A_fin, init_b = b_fin, seed = seed, init_alpha = 0., test_percentage = test_p,newdata = newdata)
dfgrid2 = result5.df

# Grid search epsilon
result4 = prob.grid(epslst = np.linspace(0.01, 5, 80), init_A = init, init_b = init_bval, seed = seed, init_alpha = 0., test_percentage =test_p,newdata = newdata)
dfgrid = result4.df