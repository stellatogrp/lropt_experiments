import os
import sys
import joblib
from joblib import Parallel, delayed
output_stream = sys.stdout
import cvxpy as cp
import numpy as np
import time
import pandas as pd
import lropt
import hydra
from utils_lcx import calc_ab_thresh

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
    pert = np.zeros((n,2))
    pert[:,0] = np.array([j*0.02 for j in range(n)])
    pert[:,1] = np.array([j*0.06 for j in range(n)])
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
        d_train = np.random.multivariate_normal(0.7*orig_mu+ 0.3*mu[i],sig[i])
        pointlist.append(d_train)
    return np.vstack(pointlist)


def create_min(x,eps,data,Gamma,datamax,n):
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

def create_max(u_set,n):
    x = cp.Variable(n)
    t = cp.Variable()
    objective = cp.Maximize(t)
    constraints =[cp.sum(x) == 1, x >= 0]
    for u in u_set:
        constraints += [x@u >= t]
    prob = lropt.RobustProblem(objective, constraints)
    return prob, x, t

def add_cut(u,v,z,data,eps,case,datamax,n,N_train):
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

def all_cuts(u,v,z,data,eps,Gamma,datamax,n,N_train):
    obj, astar, bstar = add_cut(u, v, z, data, eps, 1,datamax,n,N_train)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    obj, astar, bstar = add_cut(u, v, z, data, eps, 2,datamax,n,N_train)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    obj, astar, bstar = add_cut(u, v, z, data, eps, 3,datamax,n,N_train)
    if obj > Gamma + TOL:
        return obj, astar, bstar
    return Gamma,np.zeros(n),0


def gen_problem(Gamma,eps,data,x,datamax,n,N_train):
    prob, objective,constraints, u, v, z = create_min(x,eps,data,Gamma,datamax,n)
    prob.solve()
    obj, astar, bstar = all_cuts(u.value,v.value,z.value,data,eps,Gamma,datamax,n,N_train)
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
        obj, astar, bstar = all_cuts(u.value,v.value,z.value,data,eps,Gamma,datamax,n,N_train)
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
    return -cvar_loss, vio/u.shape[0], val/u.shape[0], -quantile_value
     
def min_max(eps,alpha,data,datamax,test,validate,seed,hydra_out_dir,n,N_train,context_val):
    Gamma = calc_ab_thresh(data, alpha, numBoots, numSamples)
    x = np.ones(n)/n
    u_set = []
    obj,uval = gen_problem(Gamma,eps,data,x,datamax,n,N_train)
    u_set.append(uval)
    prob, x, t = create_max(u_set,n)
    prob.solve()
    objnew = prob.objective.value
    outeriter = 0
    while abs(objnew - obj) >= 1e-3:
        if outeriter > max_iter_outer:
            print("max outer iters reached")
            break
        outeriter+=1
        obj,uval = gen_problem(Gamma,eps,data,x.value,datamax,n,N_train)
        u_set.append(uval)
        prob, x, t = create_max(u_set,n)
        prob.solve()
        objnew = prob.objective.value
        eval, prob_vio, test_avg, quanttest = calc_eval(x.value, t.value,test,0.1)
        eval_vali, prob_vali, vali_avg, quantvali = calc_eval(x.value, t.value,validate,0.1)
        data_df = {"outer_iter":outeriter, "context_val":context_val,'seed': seed, "alpha":alpha, "eps": eps,"test_lcx_prob": prob_vio,"test_lcx_obj":quanttest,"valid_lcx_prob": prob_vali,"valid_lcx_obj":quantvali, 'time':np.inf, "in_val": t.value, "test_avg": test_avg, "valid_avg": vali_avg, "test_lcx_cvar": eval, "valid_lcx_cvar": eval_vali }
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_val)+'_'+"vals_lcx.csv",index=False)
    return eval, prob_vio, eval_vali, prob_vali, t.value, test_avg, vali_avg,outeriter, quanttest, quantvali

def lcx_exp(cfg,hydra_out_dir,seed,initseed,sig,mu,orig_mu,n,N,N_train,context_val,context_inds,valid_inds,test_inds):
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
        eval, prob_vio,eval_vali, prob_vali, in_sample, test_avg, vali_avg,outeriter,quanttest, quantvali = min_max(eps,alpha,train,datamax,test,validate,seed,hydra_out_dir,n,N_train,context_val)
        data_df = {"outer_iter":outeriter,"context_val":context_val,'seed': seed, "alpha":alpha, "eps": eps,"test_lcx_prob": prob_vio,"test_lcx_obj":quanttest,"valid_lcx_prob": prob_vali,"valid_lcx_obj":quantvali, 'time':time.time() - start_time, "in_val": in_sample, "test_avg": test_avg, "valid_avg": vali_avg,"test_lcx_cvar": eval, "valid_lcx_cvar": eval_vali}
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_val)+'_'+"vals_lcx.csv",index=False)
    except:
        print("Training failed")
    

@hydra.main(config_path="configs",config_name = "lcx_30_2000.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory: {os.getcwd()}")
    njobs = get_n_processes(30)
    context_list = np.arange(20)
    idx = cfg.idx
    n = cfg.n_val
    N = cfg.N_val
    context_val = context_list[idx]
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
    N_train = len(context_inds[context_val])
    Parallel(n_jobs=njobs)(
        delayed(lcx_exp)(cfg,hydra_out_dir,r,initseed,sig,mu,orig_mu,n,N,N_train,context_val,context_inds,valid_inds,test_inds) for r in range(R))
    
     
if __name__ == "__main__":
    TOL = 1e-6
    max_iter = 1000
    max_iter_outer = 60
    numBoots = 10000
    numSamples= 10000
    main_func()



