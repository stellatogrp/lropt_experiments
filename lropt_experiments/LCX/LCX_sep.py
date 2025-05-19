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
import time
import torch
from sklearn import datasets
import pandas as pd
import lropt
import hydra
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import warnings
from utils import calc_ab_thresh
warnings.filterwarnings("ignore")

def get_n_processes(max_n=np.inf):
    """Get number of processes from current cps number
    Parameters
    ----------
    max_n: int
        Maximum number of processes.
    Returns
    -------
    float
        Number of processes to use.
    """

    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc

def gen_sigmu_varied(n,N = 500,seed = 0):
    np.random.seed(seed)
    sig = []
    context = []
    mu = []
    origmu = np.random.uniform(0.5,1,n)
    for i in range(N):
        F = np.random.normal(size = (n,2))
        context.append(F)
        csig = 0.2*F@(F.T)
        sig.append(csig)
        mu.append(np.random.uniform(0.5,1,n))
    return np.stack(sig), np.vstack(mu), np.stack(context), origmu

def gen_demand_varied(sig,mu,orig_mu,N,seed=399):
    pointlist = []
    np.random.seed(seed)
    for i in range(N):
        d_train = np.random.multivariate_normal(0.7*orig_mu+ 0.3*mu[i],sig[i]+0.1*np.eye(orig_mu.shape[0]))
        pointlist.append(d_train)
    return np.vstack(pointlist)




def create_min(x,eps,data,Gamma,datamax):
    u = cp.Variable(n)
    v = cp.Variable(n)
    z = cp.Variable()

    objective = cp.Minimize(u@x)
    constraints = [1 <= z, z<= 1/eps]
    constraints += [u<= datamax*z + Gamma]
    constraints += [-u<= datamax*z + Gamma ]
    constraints += [v<= datamax*z + Gamma]
    constraints += [-v<= datamax*z + Gamma ]
    prob = cp.Problem(objective,constraints)
    return prob, objective, constraints, u, v, z

def create_max(u_set):
    x = cp.Variable(n)
    t = cp.Variable()
    objective = cp.Maximize(t)
    constraints =[cp.sum(x) == 1, x >= 0]
    for u in u_set:
        constraints += [x@u >= t]
    prob = lropt.RobustProblem(objective, constraints)
    return prob, x, t

def add_cut(u,v,z,data,eps,case,datamax):
    a = cp.Variable(n)
    b = cp.Variable()
    abs_a = cp.Variable(n)
    abs_b = cp.Variable()
    t = cp.Variable(N_train)
    constraints = []
    constraints += [t>=0]
    for i in range(N_train):
        constraints += [t[i] >= z * (a@data[i, :] - b) / N_train ]
    constraints += [abs_a >= a, abs_a >= -a]
    constraints += [abs_b >= b, abs_b >= -b]
    constraints += [cp.sum(abs_a)+abs_b<=1]

    if case == 1:
        constraints += [
          a@v - (z - 1) * b >= 0,
          a@u - b >= 0,
          b >= -datamax]
        objective = cp.Maximize(a@v - (z - 1) * b + a@u - b - cp.sum(t))
    elif case == 2:
        constraints += [
          a@v - (z - 1) * b >= 0,
          a@u - b <= 0,
          (z - 1) * b<= cp.norm(v,np.inf)]
        if np.abs(z-1) <= TOL:
            constraints += [b <= datamax]
        objective = cp.Maximize(a@v - (z - 1) * b - cp.sum(t))
    elif case == 3:
        constraints += [
          a@v - (z - 1) * b <= 0,
          a@u - b >= 0,
          (z - 1) * b >= -cp.norm(v,np.inf)]
        if np.abs(z-1) <= TOL:
            constraints += [b >= -datamax]
        objective = cp.Maximize(a@u - b - cp.sum(t))
    problem = cp.Problem(objective,constraints)
    problem.solve()
    return problem.objective.value, a.value,b.value

def all_cuts(u,v,z,data,eps,Gamma,datamax):
    obj, astar, bstar = add_cut(u, v, z, data, eps, 1,datamax)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    obj, astar, bstar = add_cut(u, v, z, data, eps, 2,datamax)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    obj, astar, bstar = add_cut(u, v, z, data, eps, 3,datamax)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    return Gamma,np.zeros(n),0


def gen_problem(Gamma,eps,data,x,datamax):
    prob, objective,constraints, u, v, z = create_min(x,eps,data,Gamma,datamax)
    prob.solve()
    obj, astar, bstar = all_cuts(u.value,v.value,z.value,data,eps,Gamma,datamax)
    iter = 0
    tnew = {}
    while obj > Gamma + TOL:
        if iter > max_iter:
            print("Max iter reached")
            break
        iter += 1
        tnew[iter] = cp.Variable(2)
        constraints += [tnew[iter] >=0]
        constraints += [tnew[iter][0]>= astar@v - bstar*(z-1)]
        constraints += [tnew[iter][1]>= astar@u - bstar]
        constraints += [tnew[iter][0]+tnew[iter][1]<= z*cp.sum((cp.maximum(cp.matmul(data, astar) - bstar, 0)))/N_train + Gamma]
        prob = cp.Problem(objective,constraints)
        prob.solve()
        # print(iter,obj, prob.objective.value)
        obj, astar, bstar = all_cuts(u.value,v.value,z.value,data,eps,Gamma,datamax)
    return prob.objective.value, u.value

def calc_eval(x,t,u,eta):
    val = 0
    vio = 0
    port_values = u@x
    quantile_index = int((1-eta) * len(port_values)) 
    port_sorted = np.sort(port_values)[::-1]  # Descending sort
    quantile_value = port_sorted[quantile_index]
    port_le_quant = (port_values <= quantile_value).astype(float)
    cvar_loss = np.sum(port_values * port_le_quant) / np.sum(port_le_quant)
    for i in range(u.shape[0]):
        val_cur = -x@u[i]
        val+= val_cur
        vio += (val_cur >= t)
    return -cvar_loss, vio/u.shape[0], val/u.shape[0]
     
def min_max(eps,alpha,data,datamax,test,validate):
    Gamma = calc_ab_thresh(data, alpha, numBoots, numSamples)
    print(eps,alpha,Gamma)
    #Gamma = 0.02214348229558842 
    #(0.1,0.05)
    x = np.ones(n)/n
    u_set = []
    obj,uval = gen_problem(Gamma,eps,data,x,datamax)
    u_set.append(uval)
    prob, x, t = create_max(u_set)
    prob.solve()
    objnew = prob.objective.value
    outeriter = 0
    while abs(objnew - obj) >= 1e-3:
        if outeriter > max_iter_outer:
            print("max outer iters reached")
            break
        outeriter+=1
        obj,uval = gen_problem(Gamma,eps,data,x.value,datamax)
        u_set.append(uval)
        prob, x, t = create_max(u_set)
        prob.solve()
        objnew = prob.objective.value
    eval, prob_vio, test_avg = calc_eval(x.value, t.value,test,0.1)
    eval_vali, prob_vali, vali_avg = calc_eval(x.value, t.value,validate,0.1)
    return eval, prob_vio, eval_vali, prob_vali, t.value, test_avg, vali_avg

def lcx_exp(cfg,hydra_out_dir,seed):
    seed = initseed + 10*seed
    print(seed)
    start_time = time.time()
    data = gen_demand_varied(sig,mu,orig_mu,N,seed=seed)
    train = data[context_inds[context_val]]
    validate = data[valid_inds[context_val]]
    test = data[test_inds[context_val]]
    datamax = np.max(np.abs(train))
    eps = cfg.eps
    alpha = cfg.alpha
    try:
        eval, prob_vio,eval_vali, prob_vali, in_sample, test_avg, vali_avg = min_max(eps,alpha,train,datamax,test,validate)
        data_df = {"context_val":context_val,'seed': seed, "alpha":alpha, "eps": eps,"test_lcx_prob": prob_vio,"test_lcx_obj":eval,"valid_lcx_prob": prob_vali,"valid_lcx_obj":eval_vali, 'time':time.time() - start_time, "in_val": in_sample, "test_avg": test_avg, "valid_avg": vali_avg}
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_val)+'_'+"vals_lcx.csv",index=False)
    except:
        print("Training failed")
    

@hydra.main(config_path="/scratch/gpfs/iywang/lropt_revision/lropt_experiments/lropt_experiments/LCX/configs",config_name = "lcx1.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory: {os.getcwd()}")
    njobs = get_n_processes(30)
    Parallel(n_jobs=njobs)(
        delayed(lcx_exp)(cfg,hydra_out_dir,r) for r in range(R))
    # for r in range(R):
    #     portfolio_exp(cfg,hydra_out_dir,r)
    
     
if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_list = [10,20,30]
    context_list = np.arange(20)
    n = 30
    context_val = context_list[idx]
    # n_list[idx]
    N = 1000
    R = 10
    num_context = 20
    test_p = 0.5
    initseed = 0
    # sig, mu = gen_sigmu(n,1)
    num_reps = int(N/num_context)
    sig, mu, context, orig_mu = gen_sigmu_varied(n,num_context,seed= 0)
    sig = np.vstack([sig]*num_reps)
    mu = np.vstack([mu]*num_reps)
    context = np.vstack([context]*num_reps)
    np.random.seed(5)
    test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
    test_indices = test_valid_indices[:int((test_p)*N)]
    valid_indices = test_valid_indices[int((test_p)*N):]
    train_indices = [i for i in range(N) if i not in test_valid_indices]
    context_inds = {}
    test_inds = {}
    valid_inds = {}
    for j in range(num_context):
        context_inds[j]= [i for i in  train_indices  if j*num_reps <= i <= (j+1)*num_reps]
        test_inds[j] = [i for i in test_indices if j*num_reps <= i <= (j+1)*num_reps]
        valid_inds[j]= [i for i in valid_indices if j*num_reps <= i <= (j+1)*num_reps]

    TOL = 1e-6
    max_iter = 1000
    max_iter_outer = 100
    numBoots = 10000
    numSamples= 10000
    N_train = len(context_inds[context_val])
    main_func()



