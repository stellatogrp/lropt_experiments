import os
import sys
import joblib
from joblib import Parallel, delayed
output_stream = sys.stdout
import cvxpy as cp
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


def portfolio_exp(cfg,hydra_out_dir,seed, initseed, sig,mu,orig_mu,N,n,train_indices,context_inds,valid_inds,test_inds,context,eps_list):
    finseed = initseed + 10*seed
    print(finseed)
    data_gen = False
    while not data_gen:
        try: 
            data = gen_demand_varied(sig,mu,orig_mu,N,seed=finseed)
            train = data[train_indices]
        except Exception as e:
            finseed += 1
        else: 
            data_gen = True
    # seed = idx
    for context_v in range(20):
        train = data[context_inds[context_v]]
        u = lropt.UncertainParameter(n,
                                uncertainty_set=lropt.MRO(K=1, p=2, data=data[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]], train_data=train, train=True))
        # Formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()
        context_param = lropt.ContextParameter((n,2), data=context[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]])
        mu_param = lropt.ContextParameter(n, data=mu[context_inds[context_v]+ valid_inds[context_v] + test_inds[context_v]])

        objective = cp.Minimize(t)
        constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]
        constraints += [context_param >= -1000, mu_param >= -1000]
        eval_exp = -x @ u

        prob = lropt.RobustProblem(objective, constraints, eval_exp=eval_exp)

        # Train A and b
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
        result_grid = trainer.grid(rholst=eps_list,settings = settings)
        dfgrid = result_grid.df
        dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
        dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+str(context_v)+'_dro_grid.csv')
    return None

@hydra.main(config_path="configs",config_name = "port_dro_sep_30_2000.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    njobs = get_n_processes(30)
    R = 10
    initseed = 0
    n = cfg.n_val
    N = cfg.N_val
    num_context = 20
    test_p = 0.5
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
    eps_list=np.linspace(0.5, 5, 60)
    Parallel(n_jobs=njobs)(
        delayed(portfolio_exp)(cfg,hydra_out_dir,r,initseed, sig,mu,orig_mu,N,n,train_indices,context_inds,valid_inds,test_inds,context,eps_list) for r in range(R))
    
if __name__ == "__main__":
    main_func()

