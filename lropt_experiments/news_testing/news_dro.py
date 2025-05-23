import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import lropt
import torch
from sklearn import datasets
import pandas as pd
from omegaconf import DictConfig
import os
import sys
import joblib
from joblib import Parallel, delayed
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hydra
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":24,
    "font.family": "serif"
})

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

def gen_demand_cor(N,seed,x1, x2):
    np.random.seed(seed)
    sig = np.eye(2)
    mu = np.array((6,9))
    points_list = []
    for i in range(N):
        mu_shift = -0.4*x1[i] - 0.1*x2[i]
        newpoint = np.random.multivariate_normal(mu+mu_shift,sig)
        points_list.append(newpoint)
    return np.vstack(points_list)


def news_exp(cfg,hydra_out_dir,seed):
    seed = init_seed + seed
    data = gen_demand_cor(N,seed=seed,x1=p_data,x2=k_data)
    test_p = cfg.test_percentage
    # split dataset
    train = data[train_indices]
    init = sc.linalg.sqrtm(np.cov(train.T))
    init_bval = np.mean(train, axis=0)

    u = lropt.UncertainParameter(n,
                            uncertainty_set=lropt.MRO(K=cfg.Kval, p=2, data=data, train_data=train, train=True))
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

    trainer = lropt.Trainer(prob)
    settings = lropt.TrainerSettings()
    settings.data = data
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
    dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'dro_grid.csv')
    solvetime = 0
    try:
        prob.solve()
        solvetime = prob.solver_stats.solve_time
    except:
        print("solving failed")
    try:
        data_df = {"seed":seed,"time": solvetime}
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
    except:
        print("save failed")


@hydra.main(config_path="/scratch/gpfs/iywang/lropt_revision/lropt_experiments/lropt_experiments/news_testing/configs",config_name = "news.yaml", version_base = None)
def main_func(cfg: DictConfig):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory: {os.getcwd()}")
    njobs = get_n_processes(30)
    Parallel(n_jobs=njobs)(
        delayed(news_exp)(cfg,hydra_out_dir,r) for r in range(R))
    

if __name__ == "__main__":
    # Formulate constant
    n = 2
    init_seed = 0
    N = 2000
    #eps_list = [0.5,0.7,0.9,1,1.1,1.3,1.5,2,2.5]
    eps_list = np.linspace(0.5,2.5,40)
    k_init = np.array([2.,3.])
    R = 10
    s = 1
    # in order for scenario to make sense, generate only 20 contexts
    np.random.seed(s)
    num_context = 20
    num_reps = int(N/num_context)
    init_k_data = np.maximum(0.5,k_init + np.random.normal(0,3,(num_context,n)))
    init_p_data = init_k_data + np.maximum(0,np.random.normal(0,3,(num_context,n)))
    p_data = np.repeat(init_p_data,num_reps,axis=0)
    k_data = np.repeat(init_k_data,num_reps,axis=0)
    test_p = 0.5
    s = 5
    np.random.seed(5)
    test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
    test_indices = test_valid_indices[:int((test_p)*N)]
    valid_indices = test_valid_indices[int((test_p)*N):]
    train_indices = [i for i in range(N) if i not in test_valid_indices]
    context_inds = {}
    test_inds = {}
    for j in range(num_context):
      context_inds[j]= [i for i in  train_indices + list([*valid_indices]) if j*num_reps <= i <= (j+1)*num_reps]
      test_inds[j] = [i for i in test_indices if j*num_reps <= i <= (j+1)*num_reps]
    main_func()