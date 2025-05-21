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
    mu = np.array((6,7))
    points_list = []
    for i in range(N):
        mu_shift = -0.4*x1[i] - 0.1*x2[i]
        newpoint = np.random.multivariate_normal(mu+mu_shift,sig)
        points_list.append(newpoint)
    return np.vstack(points_list)

# uncertain data depends on the contexts
def calc_eval(x,p,k,u,t,eta):
    val = 0
    vio = 0
    vals = []
    for i in range(u.shape[0]):
        val_cur = k@x + np.max([-p[0]*x[0] - p[1]*x[1],-p[0]*x[0] - p[1]*u[i][1], -p[0]*u[i][0] - p[1]*x[1], -p[0]*u[i][0]- p[1]*u[i][1]]) 
        vals.append(val_cur)
        val+= val_cur
        vio += (val_cur >= t)
    vals = -np.array(vals)
    quantile_index = int((1-eta) * len(vals)) 
    vals_sorted = np.sort(vals)[::-1]  # Descending sort
    quantile_value = vals_sorted[quantile_index]
    vals_le_quant = (vals <= quantile_value).astype(float)
    cvar_loss = np.sum(vals * vals_le_quant) / np.sum(vals_le_quant)
    return -cvar_loss, vio/u.shape[0],val/u.shape[0]

def news_exp(cfg,hydra_out_dir,seed):
    seed = init_seed + seed
    data = gen_demand_cor(N,seed=seed,x1=p_data,x2=k_data)
    test_p = cfg.test_percentage
    # split dataset
    train = data[train_indices]
    init = sc.linalg.sqrtm(np.cov(train.T))
    init_bval = np.mean(train, axis=0)

    if cfg.eta == 0.01 and cfg.obj_scale == 1:
        context_evals = 0
        context_probs = 0
        context_objs = 0
        avg_vals = 0
        # solve for each context and average
        for j in range(num_context):
            u = lropt.UncertainParameter(n,
                                    uncertainty_set=lropt.Scenario(
                                                                data=data[context_inds[j]]))
            x_s = cp.Variable(n)
            t1 = cp.Variable()
            k1= init_k_data[j]
            p1 = init_p_data[j]
            objective = cp.Minimize(t1)
            constraints = [lropt.max_of_uncertain([-p1[0]*x_s[0] - p1[1]*x_s[1],
                                                    -p1[0]*x_s[0] - p1[1]*u[1],
                                                    -p1[0]*u[0] - p1[1]*x_s[1],
                                                    -p1[0]*u[0]- p1[1]*u[1]])
                                                    + k1@x_s <= t1]
            constraints += [x_s >= 0]

            prob_sc = lropt.RobustProblem(objective, constraints)
            prob_sc.solve()
            eval, prob_vio, avg = calc_eval(x_s.value, init_p_data[j], init_k_data[j],data[test_inds[j]],t1.value,cfg.target_eta)
            context_evals += eval
            context_probs += prob_vio
            avg_vals += avg
            context_objs += t1.value
        context_evals = context_evals/num_context
        context_probs = context_probs/num_context
        context_objs = context_objs/num_context
        context_avg = avg_vals/num_context

        nonrob_evals = 0
        nonrob_probs = 0
        nonrob_objs = 0
        avg_vals = 0
        for j in range(num_context):
            u = lropt.UncertainParameter(n,
                                    uncertainty_set=lropt.Scenario(
                                                                data=np.mean(data[context_inds[j]],axis=0).reshape(1,n)))
            x_s = cp.Variable(n)
            t1 = cp.Variable()
            k1= init_k_data[j]
            p1 = init_p_data[j]
            objective = cp.Minimize(t1)
            constraints = [lropt.max_of_uncertain([-p1[0]*x_s[0] - p1[1]*x_s[1],
                                                    -p1[0]*x_s[0] - p1[1]*u[1],
                                                    -p1[0]*u[0] - p1[1]*x_s[1],
                                                    -p1[0]*u[0]- p1[1]*u[1]])
                                                    + k1@x_s <= t1]
            constraints += [x_s >= 0]

            prob_sc = lropt.RobustProblem(objective, constraints)
            prob_sc.solve()

            eval, prob_vio,avg = calc_eval(x_s.value, init_p_data[j], init_k_data[j],data[test_inds[j]],t1.value,cfg.target_eta)
            nonrob_evals += eval
            nonrob_probs += prob_vio
            nonrob_objs += t1.value
            avg_vals += avg

        nonrob_evals = nonrob_evals / (num_context)
        nonrob_probs = nonrob_probs / (num_context)
        nonrob_objs = nonrob_objs/num_context
        nonrob_avg = avg_vals/num_context

        data_df = {'seed': seed, "a_seed":seed,"nonrob_prob": nonrob_probs, "nonrob_obj":nonrob_evals, "scenario_probs": context_probs, "scenario_obj": context_evals, "scenario_in": context_objs, "nonrob_in": nonrob_objs, "scenario_avg":context_avg, "nonrob_avg": nonrob_avg}
        single_row_df = pd.DataFrame(data_df, index=[0])
        single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals_nonrob.csv",index=False)

    # Formulate uncertainty set
    u = lropt.UncertainParameter(n,
                            uncertainty_set=lropt.Ellipsoidal(
                                                        data=data))
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


    # Train A and b
    from lropt import Trainer
    trainer = Trainer(prob)
    settings = lropt.TrainerSettings()
    settings.lr= cfg.lr
    settings.optimizer=cfg.optimizer
    settings.seed=5
    settings.init_A= init
    settings.init_b=init_bval
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
    settings.data = data
    settings.cost_func = True
    settings.target_eta = cfg.target_eta
    if cfg.eta == 0.01 and cfg.obj_scale == 1:
        # no training (steps = 1, look at initalized set)
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=False, lr=0.001,epochs = 200,knn_cov=True,n_neighbors = int(0.1*N*0.3),knn_scale = cfg.knn_mult)
        settings.num_iter = 1 
        result2 = trainer.train(settings=settings)
        A_fin2 = result2.A
        b_fin2 = result2.b
        torch.save(result2._predictor.state_dict(),hydra_out_dir+'/'+'pretrained_linear.pth')
        # untrained linear
        settings.init_A = A_fin2
        settings.init_b = b_fin2
        settings.predictor = result2._predictor
        result_grid3 = trainer.grid(rholst=eps_list,settings=settings)
        dfgrid3 = result_grid3.df
        dfgrid3 = dfgrid3.drop(columns=["z_vals","x_vals"])
        dfgrid3.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'linear_pretrained_grid.csv')

    # training
    try:
        settings.num_iter = cfg.num_iter
        settings.init_A = init
        settings.init_b = init_bval
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain = True,epochs = 20, lr = 0.001)
        settings.predictor = lropt.DeepNormalModel()
        result = trainer.train(settings=settings)
        torch.save(result._predictor.state_dict(),hydra_out_dir+'/'+str(seed)+'_'+'trained_linear.pth')
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
            data_df = {'seed': seed, 'rho':rho, "a_seed":seed, 'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'valid_obj': df_valid["Validate_cvar"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_cvar"][0], 'test_prob': df_test["Avg_prob_test"][0],"time": solvetime,"valid_cover":df_valid["Coverage_validate"][0], "test_cover": df_test["Coverage_test"][0], "valid_in": df_valid["Validate_insample"][0], "test_in": df_test["Test_insample"][0], "avg_val": df_test["Test_val"][0]}
            single_row_df = pd.DataFrame(data_df, index=[0])
            findfs.append(single_row_df)
        findfs = pd.concat(findfs)
        findfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
    except:
        None

    if cfg.eta == 0.01 and cfg.obj_scale == 1:
        # Grid search epsilon
        # mean variance set
        settings.contextual = False
        result_grid = trainer.grid(rholst=eps_list,settings=settings)
        dfgrid = result_grid.df
        dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
        dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'mean_var_grid.csv')

    
    beg1, end1 = 0, 100
    beg2, end2 = 0, 100
    plt.figure(figsize=(15, 4))
    
    if cfg.eta == 0.01 and cfg.obj_scale == 1:
        plt.plot(np.mean(np.vstack(dfgrid['Avg_prob_validate']), axis=1)[beg1:end1], np.mean(np.vstack(
            dfgrid['Validate_val']), axis=1)[beg1:end1], color="tab:blue", label=r"Mean-Var validate set", marker="v", zorder=0)
        plt.plot(np.mean(np.vstack(dfgrid3['Avg_prob_validate']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid3['Validate_val']), axis=1)[beg2:end2], color="tab:green", label="Linear pretrained validate set", marker="^", zorder=2)

        plt.plot(np.mean(np.vstack(dfgrid['Avg_prob_test']), axis=1)[beg1:end1], np.mean(np.vstack(
        dfgrid['Test_val']), axis=1)[beg1:end1], color="tab:blue", label=r"Mean-Var test set", marker="s", zorder=0)

        plt.plot(np.mean(np.vstack(dfgrid3['Avg_prob_test']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid3['Test_val']), axis=1)[beg2:end2], color="tab:green", label="Linear pretrained test set", marker="s", zorder=2)
        
        plt.scatter(context_probs,context_evals, color = "black", label="Scenario")
        plt.scatter(nonrob_probs,nonrob_evals, color = "tab:purple", marker = "s", label="Non Robust")

    plt.scatter(df_valid["Avg_prob_validate"][0],df_valid["Validate_val"][0], color="tab:orange", label="Linear trained test set", marker="^", zorder=1)
    plt.scatter(df_test["Avg_prob_test"][0],df_test["Test_val"][0], color="tab:orange", label="Linear trained validate set", marker="s", zorder=1)
    
    plt.ylabel("Objective value")
    plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
    # plt.ylim([-9, 0])
    plt.grid()

    plt.legend()
    plt.savefig(hydra_out_dir+'/'+str(seed)+'_'+"news_objective_vs_violations_"+str(cfg.eta)+".pdf", bbox_inches='tight')

    plt.figure(figsize=(15, 4))



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
    k_init = np.array([4.,5.])
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