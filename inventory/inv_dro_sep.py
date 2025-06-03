import os
import sys
import joblib
from joblib import Parallel, delayed
output_stream = sys.stdout
import cvxpy as cp
import scipy as sc
import numpy as np
import pandas as pd
import lropt
import hydra
import warnings
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

def gen_sigmu_varied(n,m,N = 500,seed = 0):
    np.random.seed(seed)
    sig = []
    context = []
    for i in range(N):
        F = np.random.normal(size = (n,m))
        context.append(F)
        csig = 0.3*F@(F.T)
        sig.append(csig)
    return np.stack(sig), np.stack(context)

def gen_demand_varied(sig,mu,d,N,seed=399):
    pointlist = []
    np.random.seed(seed)
    for i in range(N):
        d_train = np.random.multivariate_normal(d - 0.1*mu[i],sig[i])
        pointlist.append(d_train)
    return np.vstack(pointlist)

def calc_eval(u,r,y,Y,t,h,d,s,L):
    val = 0
    vio = 0
    for i in range(u.shape[0]):
        val_cur = -r@y - r@Y@u[i] + (t+h)@s
        val+= val_cur
        sum = (val_cur >= L)
        for j in range(n):
            sum += np.sum((y+Y@u[i] - s) >= 0)
            sum+= np.sum((y + Y@u[i] - u[i]) >= 0)
        vio += (sum >= 0.0001)
    return val/u.shape[0], vio/u.shape[0]

def inv_exp(cfg,hydra_out_dir,seed,idxx):
    finseed = initseed + 10*seed
    print(finseed)
    data_gen = False
    while not data_gen:
        try: 
            data = gen_demand_varied(sig,y_data,d,N,seed=finseed)
            train = data[train_indices]
        except Exception as e:
            finseed += 1
        else: 
            data_gen = True
    # seed = idxx

    for context_v in range(10):
        train = data[context_inds[context_v]]

        u = lropt.UncertainParameter(n,
                                        uncertainty_set = lropt.MRO(K=train.shape[0], p=2, data=data[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]], train_data = train, train=True))
        # formulate cvxpy variable
        L = cp.Variable()
        s = cp.Variable(n)
        y = cp.Variable(n)
        Y = cp.Variable((n,n))
        r = lropt.ContextParameter(n, data = y_data[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]]) 
        context = lropt.ContextParameter((n,m), data=context_dat[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]])           
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
        prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp)

        # Train A and b
        try:
            trainer = lropt.Trainer(prob)
            settings = lropt.TrainerSettings()
            settings.data = data[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]]
            settings.target_eta = 0.1
            settings.init_A = np.eye(n)
            settings.init_b = np.zeros(n)
            settings.seed = 5
            settings.contextual = False
            settings.test_percentage=cfg.test_percentage
            settings.validate_percentage = cfg.validate_percentage
            result_grid = trainer.grid(rholst=eps_list, settings = settings)
            dfgrid = result_grid.df
            dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
            dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_v)+'_dro_grid.csv')
        except:
            print("grid failed")
        solvetime = 0
        try:
            prob.solve()
            solvetime = prob.solver_stats.solve_time
        except:
            print("solving failed")
        try:
            data_df = {"seed":initseed+10*seed,"time": solvetime}
            single_row_df = pd.DataFrame(data_df, index=[0])
            single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_v)+"_vals.csv",index=False)
        except:
            print("save failed")
    return None

@hydra.main(config_path="/scratch/gpfs/iywang/lropt_revision/lropt_experiments/lropt_experiments/inventory_parallel/configs",config_name = "inv_dro_sep.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    njobs = get_n_processes(30)
    Parallel(n_jobs=njobs)(
        delayed(inv_exp)(cfg,hydra_out_dir,r,cfg.idx) for r in range(R))
    

if __name__ == "__main__":
    R = 10
    initseed = 0
    test_p = 0.5
    N = 1000
    n = 10
    m = 4
    np.random.seed(27)
    y_nom = np.random.uniform(2,4,n)
    y_data = y_nom
    num_context = 10
    num_reps = int(N/num_context)
    for scene in range(num_context-1):
        np.random.seed(scene)
        y_data = np.vstack([y_data,np.maximum(y_nom + np.random.normal(0,0.1,n),0)])
    np.random.seed(27)
    C = 200
    c = np.random.uniform(30,50,n)
    d = np.random.uniform(10,20,n)
    t = np.random.uniform(0.1,0.3,n)
    h = np.random.uniform(0.1,0.3,n)
    sig, context = gen_sigmu_varied(n,m,num_context,seed= 0)
    sig = np.vstack([sig]*num_reps)
    context_dat = np.vstack([context]*num_reps)
    y_data = np.vstack([y_data]*num_reps)
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
    eps_list=np.linspace(0.1, 4.5, 50)
    main_func()

