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
from utils import plot_iters, plot_coverage_all


n = 20
seed = 15
np.random.seed(seed)
dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
                20, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])/10)[:n]
y_data = np.random.dirichlet(dist, 10)
def gen_demand(n, N, seed=399):
    np.random.seed(0)
    F = np.random.normal(size = (n,2))
    # sig = np.random.uniform(0,0.9,(n,n))
    sig = 0.1*F@(F.T)
    mu = np.random.uniform(0.5,1,n)
    np.random.seed(seed)
    d_train = np.random.multivariate_normal(mu,sig, N)
    return d_train

def f_tch(t, x, y, u):
    # x is a tensor that represents the cp.Variable x.
    return t + 0.2*torch.linalg.vector_norm(x-y, 1)

def g_tch(t, x, y, u):
    # x,y,u are tensors that represent the cp.Variable x and cp.Parameter y and u.
    # The cp.Constant c is converted to a tensor
    return -x @ u.T - t

def eval_tch(t, x, y, u):
    return -x @ u.T + 0.2*torch.linalg.vector_norm(x-y, 1)


seed = 0
for N in np.flip(np.array([50,80,100,300,500,1000,1500,2000,3000,5000])):
    # seed += 1
    data = gen_demand(n,N, seed=seed)

    u = lropt.UncertainParameter(n,
                            uncertainty_set=lropt.Ellipsoidal(p=2,
                                                        data=data))
    # Formulate the Robust Problem
    x = cp.Variable(n)
    t = cp.Variable()
    y = lropt.Parameter(n, data=y_data)

    objective = cp.Minimize(t + 0.2*cp.norm(x - y, 1))
    constraints = [-x@u.T <= t, cp.sum(x) == 1, x >= 0]

    prob = lropt.RobustProblem(objective, constraints, objective_torch=f_tch, constraints_torch=[g_tch], eval_torch=eval_tch)
    test_p = 0.2
    s = 0
    #s=0,2,4,6,0
    train, test = train_test_split(data, test_size=int(
        data.shape[0]*test_p), random_state=s)
    init = np.real(sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T))))
    init_bval = -init@np.mean(train, axis=0)
    np.random.seed(15)
    initn = np.random.rand(n, n) + 0.1*init + 0.5*np.eye(n)
    init_bvaln = -initn@(np.mean(train, axis=0) - 0.3*np.ones(n))
    #iters = 5000
    # Train A and b
    result = prob.train(lr=0.01, num_iter=8000, optimizer="SGD",
                        seed=s, init_A=initn, init_b=init_bvaln, init_lam=0.5, init_mu=0.01,
                        mu_multiplier=1.001, init_alpha=0., test_percentage = test_p, save_history = False, lr_step_size = 300, lr_gamma = 0.2, position = False, random_init = True, num_random_init=5, parallel = True)
    df = result.df
    A_fin = result.A
    b_fin = result.b

    result5 = prob.grid(epslst=np.linspace(0.1, 8, 200), init_A=A_fin, init_b=b_fin, seed=s,
                        init_alpha=0., test_percentage=test_p)
    dfgrid2 = result5.df

    result4 = prob.grid(epslst=np.linspace(0.1, 8, 200), init_A=init,
                        init_b=init_bval, seed=s,
                        init_alpha=0., test_percentage=test_p)
    dfgrid = result4.df

    plot_coverage_all(dfgrid,dfgrid2,None, f"results/results4/port(N,m)_{N,n}",ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)

    plot_iters(df, result.df_test, f"results/results4/port(N,m)_{N,n}", steps = 10000,logscale = 1)

    dfgrid.to_csv(f"results/results4/gridmv_{N,n}.csv")
    dfgrid2.to_csv(f"results/results4/gridre_{N,n}.csv")
    result.df_test.to_csv(f"results/results4/retrain_{N,n}.csv")