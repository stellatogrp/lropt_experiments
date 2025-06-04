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


def portfolio_exp(cfg,hydra_out_dir,seed,initseed, sig,mu,orig_mu,N,n,train_indices,context,eps_list,num_context,context_inds,test_inds):
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

    if cfg.eta == 0.05 and cfg.obj_scale== 1:
        context_evals = 0
        context_probs = 0
        context_objs = 0
        avg_vals = 0
        quant_val = 0
        for j in range(num_context):
            u = lropt.UncertainParameter(n,
                                    uncertainty_set=lropt.Scenario(
                                                                data=data[context_inds[j]]))
            # Formulate the Robust Problem
            x_s = cp.Variable(n)
            t_s = cp.Variable()

            objective = cp.Minimize(t_s)
            constraints = [-x_s@u <= t_s, cp.sum(x_s) == 1, x_s >= 0]
            prob_context = lropt.RobustProblem(objective, constraints)
            prob_context.solve()
            eval, prob_vio, avg, quantval = calc_eval(x_s.value, t_s.value,data[test_inds[j]],cfg.target_eta)
            context_evals += eval
            context_probs += prob_vio
            avg_vals += avg
            context_objs += t_s.value
            quant_val += quantval

        
        context_evals = context_evals/num_context
        context_probs = context_probs/num_context
        context_objs = context_objs/num_context
        context_avg = avg_vals/num_context
        context_quant = quant_val/num_context

        nonrob_evals = 0
        nonrob_probs = 0
        nonrob_objs = 0
        avg_vals = 0
        quant_val = 0
        for j in range(num_context):
            u = lropt.UncertainParameter(n,
                                    uncertainty_set=lropt.Scenario(
                                                                data=np.mean(data[context_inds[j]],axis=0).reshape(1,n)))
            # Formulate the Robust Problem
            x_s = cp.Variable(n)
            t_s = cp.Variable()

            objective = cp.Minimize(t_s)
            constraints = [-x_s@u <= t_s, cp.sum(x_s) == 1, x_s >= 0]
            prob_nonrob = lropt.RobustProblem(objective, constraints)
            prob_nonrob.solve()
            eval, prob_vio, avg,quantval = calc_eval(x_s.value, t_s.value,data[test_inds[j]],cfg.target_eta)
            nonrob_evals += eval
            nonrob_probs += prob_vio
            nonrob_objs += t_s.value
            avg_vals += avg
            quant_val += quantval
        nonrob_evals = nonrob_evals / (num_context)
        nonrob_probs = nonrob_probs / (num_context)
        nonrob_objs = nonrob_objs/num_context
        nonrob_avg = avg_vals/num_context
        nonrob_quant = quant_val/num_context

        data_df = {'seed': initseed+10*seed, "a_seed":finseed,"nonrob_prob": nonrob_probs, "nonrob_obj":nonrob_quant, "scenario_probs": context_probs, "scenario_obj": context_quant, "scenario_in": context_objs, "nonrob_in": nonrob_objs, "scenario_avg":context_avg, "nonrob_avg": nonrob_avg, "scenario_cvar":context_evals, "nonrob_cvar": nonrob_evals}
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals_nonrob.csv",index=False)


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
    num_iters = cfg.num_iter
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
    settings.predictor = lropt.LinearPredictor(predict_mean = True, predict_cov = True, pretrain=True, lr=0.001,epochs = 200,knn_cov=False,n_neighbors = int(0.1*N*0.3),knn_scale = cfg.knn_mult_train)
    settings.data = data
    settings.target_eta = cfg.target_eta
    settings.avg_scale = cfg.avg_scale
    print("training start")
    try:
        result = trainer.train(settings=settings)
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
        print("compare failed")

    if cfg.eta == 0.05 and cfg.obj_scale == 1:
        settings.init_rho = cfg.init_rho
        settings.num_iter = 1
        settings.contextual = False
        result_grid = trainer.grid(rholst=eps_list, settings=settings)
        dfgrid = result_grid.df
        dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
        dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'mean_var_grid.csv')

        # untrained linear
        settings.contextual = True
        settings.initialize_predictor = True
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=False, lr=0.001,epochs = 200,knn_cov=True,n_neighbors = int(0.1*N*0.3),knn_scale = cfg.knn_mult)
        settings.num_iter = 1
        result2 = trainer.train(settings=settings)
        A_fin2 = result2.A
        b_fin2 = result2.b
        settings.init_A = A_fin2
        settings.init_b = b_fin2
        settings.predictor = result2._predictor
        result_grid3 = trainer.grid(rholst=eps_list,settings=settings)
        dfgrid3 = result_grid3.df
        dfgrid3 = dfgrid3.drop(columns=["z_vals","x_vals"])
        dfgrid3.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'linear_pretrained_grid.csv')


@hydra.main(config_path="configs",config_name = "port_30_2000.yaml", version_base = None)
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
    test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
    test_indices = test_valid_indices[:int((test_p)*N)]
    valid_indices = test_valid_indices[int((test_p)*N):]
    train_indices = [i for i in range(N) if i not in test_valid_indices]
    context_inds = {}
    test_inds = {}
    for j in range(num_context):
      context_inds[j]= [i for i in  train_indices + list([*valid_indices]) if j*num_reps <= i <= (j+1)*num_reps]
      test_inds[j] = [i for i in test_indices if j*num_reps <= i <= (j+1)*num_reps]
    eps_list= np.concat([np.logspace(-4,-1,15),np.linspace(0.11,1,20),np.linspace(1.1,3.5,25)])
    Parallel(n_jobs=njobs)(
        delayed(portfolio_exp)(cfg,hydra_out_dir,r,initseed, sig,mu,orig_mu,N,n,train_indices,context,eps_list,num_context,context_inds,test_inds) for r in range(R))
    

if __name__ == "__main__":
    main_func()

