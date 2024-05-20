import numpy as np
import pandas as pd
from math import ceil
from itertools import product
import lropt
from utils import plot_iters, plot_coverage_all
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

# from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange

import pickle

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import matplotlib.transforms as transforms

import torch

DTYPE = torch.double
SEED = 42

# experiment default settings
N_DATA = 1000
MAX_EPOCH = 100
TEST_PERC = 0.25
CORR = 0.5

# parameters of matpower case 5

d = np.array([0.0, 3.0, 3.0, 4.0, 0.0])
pmax = np.array([0.4, 1.7, 5.2, 2.0, 6.0])
pmin = np.zeros(len(pmax))
smax = np.array([4.0, 1.9, 2.2, 1.0, 1.0, 2.4])
ptdf_str  = '-0.193917 0.475895   0.348989  0.0  -0.159538;'
ptdf_str += '-0.437588  -0.258343  -0.189451  0.0  -0.36001;'
ptdf_str += '-0.368495  -0.217552  -0.159538  0.0   0.519548;'
ptdf_str += '-0.193917  -0.524105   0.348989  0.0  -0.159538;'
ptdf_str += '-0.193917  -0.524105  -0.651011  0.0  -0.159538;'
ptdf_str += '0.368495   0.217552   0.159538  0.0   0.48045'
ptdf = np.matrix(ptdf_str)
cE = np.array([14.0, 15.0, 30.0, 40.0, 10.0]) # linear cost
cE_quad = np.sqrt(cE * 0.1) # quadratic cost
cR = np.array([80., 80., 15., 30., 80.])
basemva = 100
genloc = np.array([1, 1, 3, 4, 5]) -1
windloc = np.array([3, 5]) - 1  # standard wind farm location
# windloc = np.array([3, 2]) - 1  # configuration B
w = np.array([1.0, 1.5])
w_cap = np.array([2.0, 3.0])
G = len(genloc)
D = len(windloc)
L = ptdf.shape[0]
B = ptdf.shape[1]
gen2bus = np.zeros((B,G))
for g, bus in enumerate(genloc):
    gen2bus[bus, g] = 1
wind2bus = np.zeros((B,D))
for u, bus in enumerate(windloc):
    wind2bus[bus, u] = 1


def box_robust_dcopf_problem_param(mu_init, sigma_init, demand, wind, allow_slack=False, quadratic_cost=False, gamma=0):


    # some settings
    A_base = 10
    slack_base = 10
    obj_base = basemva/10
    
    FR = 0.8 # reduction of line capacity
    
    # define mean and uncertainty of wind power injections as parameters
    mu = cp.Parameter(D, value=mu_init, name="mu")
    sigma = cp.Parameter(D, value=sigma_init, name="sigma")
    
     # define load as a parameter
    d = cp.Parameter(B, value=demand, name="demand")
    w = cp.Parameter(D, value=wind, name="wind")
        
    # main variables
    p  = cp.Variable(G, pos=True, name="p")
    rp = cp.Variable(G, pos=True, name="rp")
    rm = cp.Variable(G, pos=True, name="rm")
    A  = cp.Variable((G,D), pos=True, name="A")
    fRAMp = cp.Variable(L, pos=True, name="fRAMp")
    fRAMm = cp.Variable(L, pos=True, name="fRAMm")

    # aux. variables for robust constraints
    z = cp.Variable((2*G + 2*L,D), name="z")
    
    # aux. variables to ensure feasibility
    if allow_slack:
        slack = cp.Variable(2*G + 2*L, pos=True, name="slack")
    
    # basic det constraints
    flow = ptdf @ ((gen2bus @ p) + (wind2bus @ w) - d)
    consts = [
        cp.sum(p) + cp.sum(w) == cp.sum(d),
        p + rp <= pmax,
        p - rm >= pmin, 
        A.T @ np.ones(G) == np.ones(D)*A_base,
         flow + fRAMp == smax * FR,
        -flow + fRAMm == smax * FR
    ]

    # box support constraints
    for g in range(G):
        if allow_slack:
            consts.append((mu.T @ (-A[g,:]/A_base)) + (sigma.T @ A[g,:]/A_base) <= rp[g] + slack[g]/slack_base)
        else:
            consts.append((mu.T @ (-A[g,:]/A_base)) + (sigma.T @ A[g,:]/A_base) <= rp[g])
        if allow_slack:
            consts.append((mu.T @ (A[g,:]/A_base)) + (sigma.T @  A[g,:]/A_base) <= rm[g] + slack[g+G]/slack_base)
        else:
            consts.append((mu.T @ (A[g,:]/A_base)) + (sigma.T @  A[g,:]/A_base) <= rm[g])
    for l in range(L):
        Bl = cp.reshape(ptdf[l,:] @ (wind2bus - (gen2bus @ A/A_base)), D)
        # Bl = (ptdf[l,:] @ (wind2bus - (gen2bus @ A))).T
        if allow_slack:
            consts.append(mu.T @ Bl + (sigma.T @ z[l,:]) <= fRAMp[l] + slack[2*G+l]/slack_base)
        else:
            consts.append(mu.T @ Bl + (sigma.T @ z[l,:]) <= fRAMp[l])
        consts.append(z[l,:] >= Bl)
        consts.append(z[l,:] >= -Bl)
        if allow_slack:
            consts.append(mu.T @ -Bl + (sigma.T @ z[L+l,:]) <= fRAMm[l] + slack[2*G+L+l]/slack_base)   
        else:
            consts.append(mu.T @ -Bl + (sigma.T @ z[L+l,:]) <= fRAMm[l])
        consts.append(z[L+l,:] >= -Bl)
        consts.append(z[L+l,:] >= Bl)

    # objective
    cost_E = (cE.T @ p)
    if quadratic_cost:
        cost_E_quad = cp.sum_squares(cp.multiply(cE_quad, p))
    else:
        cost_E_quad = 0                         
    cost_R = (cR.T @ (rp + rm))
    objective = cost_E + cost_E_quad + cost_R
    
    if allow_slack:
        thevars = [p, rp, rm, A, fRAMp, fRAMm, z, slack]
    else:
        thevars = [p, rp, rm, A, fRAMp, fRAMm, z]
    x = cp.hstack([v.flatten() for v in thevars])
    regularization = gamma * cp.sum_squares(x)
    objective += regularization
    
    if allow_slack:
        penalty_slack = cp.sum(slack) * obj_base * 1e3
        objective += penalty_slack
    
    theprob = cp.Problem(cp.Minimize(objective), consts)
    
    return theprob, thevars, [d, w, mu, sigma], consts




def box_robust_dcopf_problem_param_l_max(mu_init, sigma_init, demand, wind, allow_slack=False, quadratic_cost=False, gamma=0, train = False, traindat = None, inita=None, initb=None, initeps=1, p=np.inf, MRO = False, K=1):

    
    # some settings
    A_base = 10
    slack_base = 10
    obj_base = basemva/10
    
    FR = 0.8 # reduction of line capacity

    # define uncertain parameter and support 
    # Dsupp = np.vstack([np.eye(D),-np.eye(D)])
    # dsupp = np.hstack([np.array(mu_init) + np.array(sigma_init), -np.array(mu_init) + np.array(sigma_init)])
    # u = lropt.UncertainParameter(D,
    #                             uncertainty_set=lropt.Ellipsoidal(p=np.inf,
    #                                                         a=np.eye(D), b=-mu_init, rho = np.max(sigma_init), c = Dsupp, d = dsupp))
    if train: 
        u = lropt.UncertainParameter(D,
                                    uncertainty_set=lropt.Ellipsoidal(p=p,a=np.diag(np.array(sigma_init)), b=mu_init,
                                                                data=traindat))
        
        # define mean and uncertainty of wind power injections as parameters
        # mu = cp.Parameter(D, value=mu_init, name="mu")
        # sigma = cp.Parameter(D, value=sigma_init, name="sigma")
        
        # define load as a parameter
        d = lropt.Parameter(B, data=demand, name="demand")
        w = lropt.Parameter(D, data=wind, name="wind")
    else: 
        u = lropt.UncertainParameter(D,
                                    uncertainty_set=lropt.Ellipsoidal(p=p,
                                                                a=np.diag(np.array(sigma_init)), b=mu_init, rho = 1))
        d = cp.Parameter(B, value=demand, name="demand")
        w = cp.Parameter(D, value=wind, name="wind")
    if inita is not None:
        u = lropt.UncertainParameter(D,
                                    uncertainty_set=lropt.Ellipsoidal(p=p,
                                                                a=inita, b=initb, rho = initeps))
        
        # define mean and uncertainty of wind power injections as parameters
        # mu = cp.Parameter(D, value=mu_init, name="mu")
        # sigma = cp.Parameter(D, value=sigma_init, name="sigma")
        
        # define load as a parameter
    if MRO:
        u = lropt.UncertainParameter(D,
                                    uncertainty_set=lropt.MRO(K=K, p=p,
                                                                 rho = initeps, train = train, data=traindat))
        
        
    # main variables
    p  = cp.Variable(G, pos=True, name="p")
    rp = cp.Variable(G, pos=True, name="rp")
    rm = cp.Variable(G, pos=True, name="rm")
    A  = cp.Variable((G,D), pos=True, name="A")
    fRAMp = cp.Variable(L, pos=True, name="fRAMp")
    fRAMm = cp.Variable(L, pos=True, name="fRAMm")

    # aux. variables for robust constraints
    # z = cp.Variable((2*G + 2*L,D), name="z")
    
    # aux. variables to ensure feasibility
    if allow_slack:
        slack = cp.Variable(2*G + 2*L, pos=True, name="slack")
    
    # basic det constraints
    flow = ptdf @ ((gen2bus @ p) + (wind2bus @ w) - d)
    consts = [
        cp.sum(p) + cp.sum(w) == cp.sum(d),
        p + rp <= pmax,
        p - rm >= pmin, 
        A.T @ np.ones(G) == np.ones(D)*A_base,
         flow + fRAMp == smax * FR,
        -flow + fRAMm == smax * FR
    ]
    maxcons = []
    # box support constraints
    for g in range(G):
        if allow_slack:
            maxcons.append((-A[g,:]/A_base)@u - rp[g] - slack[g]/slack_base )
        else:
            maxcons.append(-A[g,:]@u - rp[g])
        if allow_slack:
            maxcons.append(((A[g,:]/A_base))@u- rm[g] - slack[g+G]/slack_base)
        else:
            maxcons.append(((A[g,:]/A_base))@u - rm[g])
    for l in range(L):
        Bl = cp.reshape(ptdf[l,:] @ (wind2bus - (gen2bus @ A/A_base)), D)
        if allow_slack:
            maxcons.append(Bl@u- fRAMp[l] - slack[2*G+l]/slack_base)
        else:
            maxcons.append(Bl@u - fRAMp[l])
        if allow_slack:
            maxcons.append(-Bl@u - fRAMm[l] - slack[2*G+L+l]/slack_base) 
        else:
            maxcons.append(-Bl@u -  fRAMm[l])
    
    consts.append(cp.maximum(*maxcons)<=0)

    # objective
    cost_E = (cE.T @ p)
    if quadratic_cost:
        cost_E_quad = cp.sum_squares(cp.multiply(cE_quad, p))
    else:
        cost_E_quad = 0                         
    cost_R = (cR.T @ (rp + rm))
    objective = cost_E + cost_E_quad + cost_R
    
    if allow_slack:
        thevars = [p, rp, rm, A, fRAMp, fRAMm,slack]
    else:
        thevars = [p, rp, rm, A, fRAMp, fRAMm]
    x = cp.hstack([v.flatten() for v in thevars])
    regularization = gamma * cp.sum_squares(x)
    objective += regularization
    
    if allow_slack:
        penalty_slack = cp.sum(slack) * obj_base * 1e3
        objective += penalty_slack
    
    theprob = lropt.RobustProblem(cp.Minimize(objective), consts, eval_exp = objective)
    
    return theprob, thevars, [d, w], consts

def create_historical_data(w_fcst, N=1000, SEED=42, metadata=False, corr=0.1, rel_sigma=[0.15, 0.15]):
    mu = np.zeros(D)
    rel_sigma = np.array(rel_sigma)
    correlation = np.matrix([[1.0, corr],[corr, 1.0]])
    sigma = w_fcst * rel_sigma
    Sigma = np.diag(sigma)*correlation*np.diag(sigma)
    # sample
    # np.random.seed(seed=SEED)
    hist_data = np.random.multivariate_normal(mu, Sigma, size=N)
    # truncate
    for j in range(D):
        hist_data[(hist_data[:,j] >= w_cap[j] - w_fcst[j]),j] = w_cap[j] - w_fcst[j]
        hist_data[(hist_data[:,j] <= -w_fcst[j]),j] = -w_fcst[j]
    if metadata:
        return hist_data, mu, Sigma
    else:
        return hist_data
    


TEST_PERC = 0.25
CORR = 0.5

# reset randomness
np.random.seed(seed=10)

# some other settings
d_range = [0.5, 1.1]
w_range = [0.5, 1.1]

# define bins
nbins = 10
bins = [np.linspace(w_range[0]*w[i], w_range[1]*w[i], nbins+1) for i in range(D)]

# create a large set of forecast errors from different wind scenarios
N_samples = 100
train_errors_in_bins = [[[] for bi in range(nbins+3)] for i in range(D)]
all_errors = []
for i in trange(N_samples):
    d_scenario_np = np.random.uniform(*d_range, B) * d
    w_scenario_np = np.random.uniform(*w_range, D) * w
    cur_data = create_historical_data(w_scenario_np, N=1, corr=CORR)[0]
    for i in range(D):
        cur_bin = np.digitize(w_scenario_np[i], bins[i])
        train_errors_in_bins[i][cur_bin].append(cur_data[i])
    all_errors.append(cur_data)
all_errors = np.vstack(all_errors)

train, test = train_test_split(all_errors, test_size=int(all_errors.shape[0]*TEST_PERC), random_state=SEED)

train_data = torch.tensor(train, dtype=DTYPE)
test_data = torch.tensor(test, dtype=DTYPE)

# init based on stdv
mu_init = np.mean(train, axis=0)
sigma_init =np.std(train, axis=0)/2

# percentile-based set paramters
perc= 10 # in percent
percupper = np.percentile(train, 100-perc, axis=0)
perclower = np.percentile(train, perc, axis=0)
mu_base_perc = (percupper + perclower) / 2
sigma_base_perc = mu_init + ((percupper - perclower) / 2)

# test feasibility for a few random scenarios
# can change range 0.8
d_range = [0.5, 1.1]
w_range = [0.5, 1.1]
d_scenario = np.random.uniform(*d_range, B) * d
w_scenario = np.random.uniform(*w_range, D) * w

# Generate a batch of scenarios for d and w
def gen_dw_data(num, d,w, seed=SEED):
    d_dat = d 
    w_dat = w
    for i in range(num-1):
        d_scenario = np.random.uniform(*d_range, B) * d
        w_scenario = np.random.uniform(*w_range, D) * w
        d_dat = np.vstack([d_dat,d_scenario])
        w_dat = np.vstack([w_dat, w_scenario])
    return d_dat, w_dat

d_dat, w_dat = gen_dw_data(int(N_samples*(1-TEST_PERC)),d,w)

prob_lroptmax, theparamslroptmax, _,_ = box_robust_dcopf_problem_param_l_max(mu_init, sigma_init, d, w, allow_slack=False, quadratic_cost=True)
prob_lroptmax.solve(solver="CLARABEL")

prob_lropt, thevarslropt, theparamslropt, _ = box_robust_dcopf_problem_param_l_max(mu_init, sigma_init, d_dat, w_dat, allow_slack=False, quadratic_cost=True, train=True, traindat=train, p=2, MRO=True, K=5, initeps=0.1)

init_aval = np.diag(np.array(sigma_init))
init_bval = mu_init
init_aval = np.eye(2)
init_bval = np.zeros(2)
result = prob_lropt.train(init_eps = 1000.0, lr = 0.0000001,num_iter=5, optimizer = "SGD", seed = SEED+1, init_A =init_aval, init_b = init_bval, init_lam = 50, init_mu = 10, mu_multiplier=1.01, init_alpha = -0.0, test_percentage = TEST_PERC, save_history = False, lr_step_size = 100, lr_gamma = 0.5, position = False, random_init = False, num_random_init=5, parallel = False, kappa=0., eta=0.3, contextual=True)

result_reshape = prob_lropt.grid(epslst = np.linspace(0.0001,5, 10), init_A = result.A, init_b = result.b, seed = SEED, init_alpha = 0., test_percentage = 
TEST_PERC, contextual = True, linear = result._linear)
dfgrid2 = result_reshape.df