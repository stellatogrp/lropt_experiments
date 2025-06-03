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

def calc_eval(x,t,u):
    val = 0
    vio = 0
    for i in range(u.shape[0]):
        val_cur = -x@u[i]
        val+= val_cur
        vio += (val_cur >= t)
    return val/u.shape[0], vio/u.shape[0]


def portfolio_exp(cfg,hydra_out_dir,seed):
    finseed = initseed + 10*seed
    print(finseed)
    data_gen = False
    while not data_gen:
        try: 
            data = gen_demand_varied(sig,mu,orig_mu,N,seed=finseed)
            train = data[train_indices]
            init = sc.linalg.sqrtm(np.cov(train.T))
            init_bval = np.mean(train, axis=0)
        except Exception as e:
            finseed += 1
        else: 
            data_gen = True

    u = lropt.UncertainParameter(n,
                            uncertainty_set=lropt.Ellipsoidal(p=2,
                                                        data=data))
    # Formulate the Robust Problem
    x = cp.Variable(n)
    t = cp.Variable()
    context_param = lropt.ContextParameter((n,2), data=context)
    mu_param = lropt.ContextParameter(n, data=mu)

    objective = cp.Minimize(t)
    constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]
    constraints += [context_param >= -1000, mu_param >= -1000]
    eval_exp = -x @ u

    prob = lropt.RobustProblem(objective, constraints, eval_exp=eval_exp)

    # Train A and b
    trainer = lropt.Trainer(prob)
    settings = lropt.TrainerSettings()
    settings.lr= cfg.lr
    settings.optimizer=cfg.optimizer
    settings.seed=5
    settings.init_A= init
    settings.init_b= init_bval
    settings.init_rho = cfg.init_rho
    settings.init_lam= cfg.init_lam
    settings.init_mu= cfg.init_mu
    settings.mu_multiplier= cfg.mu_multiplier
    settings.test_percentage = cfg.test_percentage
    settings.save_history = cfg.save_history
    settings.lr_step_size = cfg.lr_step_size
    settings.lr_gamma = cfg.lr_gamma
    settings.random_init = cfg.random_init
    settings.parallel = cfg.parallel
    settings.kappa = cfg.kappa
    settings.contextual = cfg.contextual
    settings.batch_percentage = cfg.batch_percentage
    settings.eta= cfg.eta
    settings.obj_scale = cfg.obj_scale
    settings.max_iter_line_search = cfg.max_iter_line_search
    settings.line_search = cfg.line_search
    settings.max_batch_size = cfg.max_batch_size
    settings.batch_percentage = cfg.batch_percentage
    settings.validate_percentage = cfg.validate_percentage
    settings.test_frequency = cfg.test_frequency
    settings.validate_frequency = cfg.validate_frequency
    settings.initialize_predictor = cfg.initialize_predictor
    settings.num_iter = cfg.num_iter
    settings.coverage_gamma = cfg.gam_scale
    settings.predictor = lropt.DeepNormalModel()
    settings.data = data
    settings.constrain_cvar = False
    settings.target_eta = cfg.target_eta
    try: 
        result = trainer.train(settings=settings)
        print("Training complete")
    except:
        print("training failed")
    solvetime = 0
    try:
        prob.solve()
        solvetime = prob.solver_stats.solve_time
    except:
        print("solving failed")
    try:
        findfs = []
        for rho in eps_list:
            df_valid, df_test = trainer.compare_predictors(settings=settings,predictors_list = [result.predictor], rho_list=[rho*result.rho])
            data_df = {'seed': initseed+10*seed, 'rho':rho, "a_seed":finseed, 'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'valid_obj': df_valid["Validate_worst"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_worst"][0], 'test_prob': df_test["Avg_prob_test"][0],"time": solvetime,"valid_cover":df_valid["Coverage_validate"][0], "test_cover": df_test["Coverage_test"][0], "valid_in": df_valid["Validate_insample"][0], "test_in": df_test["Test_insample"][0], "valid_avg": df_valid["Validate_val"][0],"avg_val": df_test["Test_val"][0],'valid_cvar': df_valid["Validate_cvar"][0], 'test_cvar': df_test["Test_cvar"][0]}
            single_row_df = pd.DataFrame(data_df, index=[0])
            findfs.append(single_row_df)
        findfs = pd.concat(findfs)
        findfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
    except:
        None

@hydra.main(config_path="/scratch/gpfs/iywang/lropt_revision/lropt_experiments/lropt_experiments/port_parallel/configs",config_name = "port_delage.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    njobs = get_n_processes(30)
    Parallel(n_jobs=njobs)(
        delayed(portfolio_exp)(cfg,hydra_out_dir,r) for r in range(R))
    

if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    seed_list = [0,0,0]
    n_list = [10,20,30]
    R = 10
    initseed = seed_list[idx]
    n = n_list[idx]
    N = 2000
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
    for j in range(num_context):
      context_inds[j]= [i for i in  train_indices + list([*valid_indices]) if j*num_reps <= i <= (j+1)*num_reps]
      test_inds[j] = [i for i in test_indices if j*num_reps <= i <= (j+1)*num_reps]
    eps_list= np.array([1])
    main_func()

