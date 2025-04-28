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
import torch
from sklearn import datasets
import pandas as pd
import lropt
import hydra
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import warnings
warnings.filterwarnings("ignore")

def gen_sigmu_varied(n,N = 500,seed = 0):
    np.random.seed(seed)
    sig = []
    context = []
    mu = []
    origmu = np.random.uniform(0.5,1,n)
    for i in range(N):
        F = np.random.normal(size = (n,2))
        context.append(F)
        csig = 0.1*F@(F.T)
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

@hydra.main(config_path="/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/portfolio_exp/configs",config_name = "port.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory: {os.getcwd()}")
    seed = cfg.seed
    data_gen = False
    while not data_gen:
        try: 
            data = gen_demand_varied(sig,mu,orig_mu,N,seed=seed)
            train = data[train_indices]
            init = sc.linalg.sqrtm(np.cov(train.T)+0.001*np.eye(n))
            init_bval = np.mean(train, axis=0)
        except Exception as e:
            seed += 1
        else: 
            data_gen = True

    context_evals = 0
    context_probs = 0
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
        eval, prob_vio = calc_eval(x_s.value, t_s.value,data[test_inds[j]])
        context_evals += eval
        context_probs += prob_vio
    context_evals = context_evals/num_context
    context_probs = context_probs/num_context

    nonrob_evals = 0
    nonrob_probs = 0
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
        eval, prob_vio = calc_eval(x_s.value, t_s.value,data[test_inds[j]])
        nonrob_evals += eval
        nonrob_probs += prob_vio
    nonrob_evals = nonrob_evals / (num_context)
    nonrob_probs = nonrob_probs / (num_context)


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
    settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=True, lr=0.001,epochs = 200)
    result = trainer.train(settings=settings)
    df = result.df
    A_fin = result.A
    b_fin = result.b
    torch.save(result._predictor.state_dict(),hydra_out_dir+'/'+'trained_linear.pth')

    result_grid4 = trainer.grid(rholst=eps_list,init_A=A_fin, init_b=b_fin, seed=5,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, predictor = result._predictor)
    dfgrid4 = result_grid4.df
    dfgrid4 = dfgrid4.drop(columns=["z_vals","x_vals"])
    dfgrid4.to_csv(hydra_out_dir+'/'+'linear_trained_grid.csv')
    
    def plot_iters(dftrain,dftest, title, steps=2000, logscale=True,kappa=0):
        plt.rcParams.update({
            "text.usetex": True,

            "font.size": 22,
            "font.family": "serif"
        })
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
        ax1.plot(dftrain["Violations_train"][:steps],
                label="In-sample empirical CVaR", linestyle="--")
        ax1.plot(np.arange(0,num_iters, settings.validate_frequency),dftest["Violations_validate"][:steps],
                label="out-of-sample empirical CVaR", linestyle="--")

        ax1.set_xlabel("Iterations")
        ax1.hlines(xmin=0, xmax=dftrain["Violations_train"][:steps].shape[0],
                y=kappa, linestyles="--", color="black", label=f"Target threshold: {kappa}")
        ax1.legend()
        ax2.plot(dftrain["Train_val"][:steps], label="In-sample objective value")
        ax2.plot(np.arange(0,num_iters, settings.validate_frequency),dftest["Validate_val"][:steps], label="Out-of-sample objective value")

        ax2.set_xlabel("Iterations")
        ax2.ticklabel_format(style="sci", axis='y',
                            scilimits=(0, 0), useMathText=True)
        ax2.legend()
        if logscale:
            ax1.set_xscale("log")
            ax2.set_xscale("log")
        plt.savefig(hydra_out_dir+'/'+title+"_iters.pdf", bbox_inches='tight')

    df_valid, df_test = trainer.compare_predictors(settings=settings,predictors_list = [result.predictor], rho_list=[result.rho])

    data_df = {'seed': cfg.seed, "a_seed":seed, 'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'rho': result.rho, 'valid_obj': df_valid["Validate_val"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_val"][0], 'test_prob': df_test["Avg_prob_test"][0],"nonrob_prob": nonrob_probs, "nonrob_obj":nonrob_evals, "scenario_probs": context_probs, "scenario_obj": context_evals}

    single_row_df = pd.DataFrame(data_df, index=[0])
    single_row_df.to_csv(hydra_out_dir+'/'+"vals.csv",index=False)

    if cfg.eta == 0.20:
        settings.init_rho = cfg.init_rho
        settings.num_iter = 1
        settings.initialize_predictor = True
        result_grid = trainer.grid(rholst=eps_list, init_A=init,
                            init_b=init_bval, seed=5,
                            init_alpha=0., test_percentage=test_p, quantiles = (0.3, 0.7))
        dfgrid = result_grid.df
        dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
        dfgrid.to_csv(hydra_out_dir+'/'+'mean_var_grid.csv')


        # untrained linear
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=True, lr=0.001,epochs = 200)
        settings.num_iter = 1
        result2 = trainer.train(settings=settings)
        A_fin2 = result2.A
        b_fin2 = result2.b
        result_grid3 = trainer.grid(rholst=eps_list,init_A=A_fin2, init_b=b_fin2, seed=5,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, predictor = result2._predictor)
        dfgrid3 = result_grid3.df
        dfgrid3 = dfgrid3.drop(columns=["z_vals","x_vals"])
        dfgrid3.to_csv(hydra_out_dir+'/'+'linear_untrained_grid.csv')
        torch.save(result2._predictor.state_dict(),hydra_out_dir+'/'+'pretrained_linear.pth')

    plot_iters(result.df,result.df_validate, steps=num_iters, title="training_"+str(cfg.eta),kappa=settings.kappa)

    beg1, end1 = 0, 100
    beg2, end2 = 0, 100
    plt.figure(figsize=(15, 4))
    
    if cfg.eta == 0.20:
        plt.plot(np.mean(np.vstack(dfgrid['Avg_prob_validate']), axis=1)[beg1:end1], np.mean(np.vstack(
            dfgrid['Validate_val']), axis=1)[beg1:end1], color="tab:blue", label=r"Mean-Var validate set", marker="v", zorder=0)
        plt.plot(np.mean(np.vstack(dfgrid3['Avg_prob_validate']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid3['Validate_val']), axis=1)[beg2:end2], color="tab:green", label="Linear pretrained validate set", marker="^", zorder=2)

        plt.plot(np.mean(np.vstack(dfgrid['Avg_prob_test']), axis=1)[beg1:end1], np.mean(np.vstack(
        dfgrid['Test_val']), axis=1)[beg1:end1], color="tab:blue", label=r"Mean-Var test set", marker="s", zorder=0)

        plt.plot(np.mean(np.vstack(dfgrid3['Avg_prob_test']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid3['Test_val']), axis=1)[beg2:end2], color="tab:green", label="Linear pretrained test set", marker="s", zorder=2)
        
    plt.scatter(df_valid["Avg_prob_validate"][0],df_valid["Validate_val"][0], color="tab:orange", label="Linear trained validate set", marker="^", zorder=1)
    plt.scatter(df_test["Avg_prob_test"][0],df_test["Test_val"][0], color="tab:orange", label="Linear trained test set", marker="s", zorder=1)

    plt.plot(np.mean(np.vstack(dfgrid4['Avg_prob_test']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid4['Test_val']), axis=1)[beg2:end2], color="tab:red", label="Linear trained grid test", marker="s", zorder=2)
    plt.plot(np.mean(np.vstack(dfgrid4['Avg_prob_validate']), axis=1)[beg2:end2], np.mean(np.vstack(
        dfgrid4['Validate_val']), axis=1)[beg2:end2], color="tab:red", label="Linear trained grid validate", marker="^", zorder=2)

    plt.ylabel("Objective value")
    plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
    # plt.ylim([-9, 0])
    plt.grid()
    plt.scatter(context_probs,context_evals, color = "black", label="Scenario")
    plt.scatter(nonrob_probs,nonrob_evals, color = "tab:purple", marker = "s", label="Non Robust")
    plt.legend()
    plt.savefig(hydra_out_dir+'/'+"port_objective_vs_violations_"+str(cfg.eta)+".pdf", bbox_inches='tight')

    plt.figure(figsize=(15, 4))

if __name__ == "__main__":
    foldername = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/portfolio_exp/"
    N =2000
    num_context = 20
    R = 1
    n = 10
    test_p = 0.5
    # sig, mu = gen_sigmu(n,1)
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
      context_inds[j]= [i for i in train_indices if j*num_reps <= i <= (j+1)*num_reps]
      test_inds[j] = [i for i in test_valid_indices if j*num_reps <= i <= (j+1)*num_reps]
    eps_list=np.linspace(0.00001, 10, 200)
    main_func()

