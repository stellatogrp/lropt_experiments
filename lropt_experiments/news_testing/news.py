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

def gen_demand_cor(N,seed,x1, x2):
    np.random.seed(seed)
    sig = np.eye(2)
    mu = np.array((6,7))
    points_list = []
    for i in range(N):
        mu_shift = -0.4*x1 - 0.1*x2
        newpoint = np.random.multivariate_normal(mu+mu_shift,sig)
        points_list.append(newpoint)
    return np.vstack(points_list)

# uncertain data depends on the contexts
def calc_eval(x,p,k,u,t):
    val = 0
    vio = 0
    for i in range(u.shape[0]):
        val_cur = k@x + np.max([-p[0]*x[0] - p[1]*x[1],-p[0]*x[0] - p[1]*u[i][1], -p[0]*u[i][0] - p[1]*x[1], -p[0]*u[i][0]- p[1]*u[i][1]]) 
        val+= val_cur
        vio += (val_cur >= t)
    return val/u.shape[0], vio/u.shape[0]

@hydra.main(config_path="/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/news_testing/configs",config_name = "news.yaml", version_base = None)
def main_func(cfg: DictConfig):
    import lropt
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory: {os.getcwd()}")
    seed = cfg.seed
    init_data = [gen_demand_cor(num_reps,seed=seed,x1=init_p_data[i],x2=init_k_data[i]) for i in range(num_context)] 
    data = np.vstack(init_data)
    test_p = cfg.test_percentage
    # split dataset
    test_indices = np.random.choice(num_reps,int((test_p)*num_reps), replace=False)
    # test_indices = testv_indices[:int((test_p)*num_reps)]
    # valid_indices = testv_indices[int((test_p)*num_reps):]
    train_indices = [i for i in range(num_reps) if i not in test_indices]

    train = np.array([init_data[j][i] for i in train_indices for j in range(num_context)])
    test = np.array([init_data[j][i] for i in test_indices for j in range(num_context)])
    # valid = np.array([init_data[j][i] for i in test_indices for j in range(num_context)])

    context_evals = 0
    context_probs = 0
    # solve for each context and average
    for context in range(num_context):
        u = lropt.UncertainParameter(n,
                                uncertainty_set=lropt.Scenario(
                                                            data=init_data[context][train_indices]))
        x_s = cp.Variable(n)
        t1 = cp.Variable()
        k1= init_k_data[context]
        p1 = init_p_data[context]
        objective = cp.Minimize(t1)
        constraints = [lropt.max_of_uncertain([-p1[0]*x_s[0] - p1[1]*x_s[1],
                                                -p1[0]*x_s[0] - p1[1]*u[1],
                                                -p1[0]*u[0] - p1[1]*x_s[1],
                                                -p1[0]*u[0]- p1[1]*u[1]])
                                                + k1@x_s <= t1]
        constraints += [x_s >= 0]

        prob_sc = lropt.RobustProblem(objective, constraints)
        prob_sc.solve()
        eval, prob_vio = calc_eval(x_s.value, init_p_data[context], init_k_data[context],init_data[context][test_indices],t1.value)
        context_evals += eval
        context_probs += prob_vio
    context_evals = context_evals/num_context
    context_probs = context_probs/num_context

    nonrob_evals = 0
    nonrob_probs = 0
    for context in range(num_context):
        u = lropt.UncertainParameter(n,
                                uncertainty_set=lropt.Scenario(
                                                            data=np.mean(init_data[context][train_indices],axis=0).reshape(1,2)))
        x_s = cp.Variable(n)
        t1 = cp.Variable()
        k1= init_k_data[context]
        p1 = init_p_data[context]
        objective = cp.Minimize(t1)
        constraints = [lropt.max_of_uncertain([-p1[0]*x_s[0] - p1[1]*x_s[1],
                                                -p1[0]*x_s[0] - p1[1]*u[1],
                                                -p1[0]*u[0] - p1[1]*x_s[1],
                                                -p1[0]*u[0]- p1[1]*u[1]])
                                                + k1@x_s <= t1]
        constraints += [x_s >= 0]

        prob_sc = lropt.RobustProblem(objective, constraints)
        prob_sc.solve()

        eval, prob_vio = calc_eval(x_s.value, init_p_data[context], init_k_data[context],init_data[context][test_indices],t1.value)
        nonrob_evals += eval
        nonrob_probs += prob_vio
    nonrob_evals = nonrob_evals / (num_context)
    nonrob_probs = nonrob_probs / (num_context)

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


    num_iters = cfg.num_iter
    initn = sc.linalg.sqrtm(np.cov(train.T))
    init_bvaln = np.mean(train, axis=0)
    # Train A and b
    from lropt import Trainer
    trainer = Trainer(prob)
    settings = lropt.TrainerSettings()
    settings.lr= cfg.lr
    settings.optimizer=cfg.optimizer
    settings.seed=5
    settings.init_A= initn
    settings.init_b=init_bvaln
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

    if cfg.eta == 0.20:
        # no training (steps = 1, look at initalized set)
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain = True,epochs = 20, lr = 0.001)
        settings.num_iter = 1 
        result2 = trainer.train(settings=settings)
        A_fin2 = result2.A
        b_fin2 = result2.b
        torch.save(result2._predictor.state_dict(),hydra_out_dir+'/'+'pretrained_linear.pth')
        # untrained linear
        result_grid3 = trainer.grid(rholst=eps_list,init_A=A_fin2, init_b=b_fin2, seed=s,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, predictor = result2._predictor)
        dfgrid3 = result_grid3.df
        dfgrid3 = dfgrid3.drop(columns=["z_vals","x_vals"])
        dfgrid3.to_csv(hydra_out_dir+'/'+'linear_untrained_grid.csv')

    # training
    settings.num_iter = cfg.num_iter
    settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain = True,epochs = 20, lr = 0.001)
    result = trainer.train(settings=settings)
    df = result.df
    df = df.drop(columns=["grad"])
    df.to_csv(hydra_out_dir+'/'+'linear_train.csv')

    dfval = result.df_validate
    dfval = dfval.drop(columns=["z_vals","x_vals"])
    dfval.to_csv(hydra_out_dir+'/'+'linear_validate.csv')
    torch.save(result._predictor.state_dict(),hydra_out_dir+'/'+'trained_linear.pth')

    plot_iters(result.df,result.df_validate, steps=num_iters, title="training_"+str(cfg.eta),kappa=settings.kappa)

    if cfg.eta == 0.20:
        # Grid search epsilon
        # mean variance set
        result_grid = trainer.grid(rholst=eps_list, init_A=initn,
                            init_b=init_bvaln, seed=s,
                            init_alpha=0., test_percentage=test_p, quantiles = (0.3, 0.7))
        dfgrid = result_grid.df
        dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
        dfgrid.to_csv(hydra_out_dir+'/'+'mean_var_grid.csv')

    df_valid, df_test = trainer.compare_predictors(settings=settings,predictors_list = [result.predictor], rho_list=[result.rho])

    data_df = {'seed': cfg.seed,'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'rho': result.rho, 'valid_obj': df_valid["Validate_val"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_val"][0], 'test_prob': df_test["Avg_prob_test"][0],"nonrob_prob": nonrob_probs, "nonrob_obj":nonrob_evals, "scenario_probs": context_probs, "scenario_obj": context_evals}

    single_row_df = pd.DataFrame(data_df, index=[0])
    single_row_df.to_csv(hydra_out_dir+'/'+"vals.csv",index=False)
    
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
        
    plt.scatter(df_valid["Avg_prob_validate"][0],df_valid["Validate_val"][0], color="tab:orange", label="Linear trained test set", marker="^", zorder=1)
    plt.scatter(df_test["Avg_prob_test"][0],df_test["Test_val"][0], color="tab:orange", label="Linear trained validate set", marker="s", zorder=1)
    
    plt.ylabel("Objective value")
    plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
    # plt.ylim([-9, 0])
    plt.grid()
    plt.scatter(context_probs,context_evals, color = "black", label="Scenario")
    plt.scatter(nonrob_probs,nonrob_evals, color = "tab:purple", marker = "s", label="Non Robust")
    plt.legend()
    plt.savefig(hydra_out_dir+'/'+"news_objective_vs_violations_"+str(cfg.eta)+".pdf", bbox_inches='tight')

    plt.figure(figsize=(15, 4))



if __name__ == "__main__":
    # Formulate constant
    n = 2
    N = 2000
    #eps_list = [0.5,0.7,0.9,1,1.1,1.3,1.5,2,2.5]
    eps_list = np.linspace(0.5,2.5,50)
    k_init = np.array([4.,5.])

    s = 1
    # in order for scenario to make sense, generate only 20 contexts
    np.random.seed(s)
    num_context = 20
    num_reps = int(N/num_context)
    init_k_data = np.maximum(0.5,k_init + np.random.normal(0,3,(num_context,n)))
    init_p_data = init_k_data + np.maximum(0,np.random.normal(0,3,(num_context,n)))
    p_data = np.repeat(init_p_data,num_reps,axis=0)
    k_data = np.repeat(init_k_data,num_reps,axis=0)
    main_func()