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

def inv_exp(cfg,hydra_out_dir,seed):
    finseed = initseed + 10*seed
    print(finseed)
    try: 
        data_gen = False
        while not data_gen:
            try: 
                data = gen_demand_varied(sig,y_data,d,N,seed=finseed)
                train = data[train_indices]
                init = sc.linalg.sqrtm(np.cov(train.T)+0.001*np.eye(n))
                init_bval = np.mean(train, axis=0)
            except Exception as e:
                finseed += 1
            else: 
                data_gen = True

        if cfg.eta == 0.10 and cfg.obj_scale==0.5:
            context_evals = 0
            context_probs = 0
            for j in range(num_context):
                u = lropt.UncertainParameter(n,
                                        uncertainty_set=lropt.Scenario(
                                                                    data=data[context_inds[j]]))
                # Formulate the Robust Problem
                L = cp.Variable()
                s = cp.Variable(n)
                y = cp.Variable(n)
                Y = cp.Variable((n,n))
                r = y_data[j]        
                Y_r = cp.Variable(n)
                # formulate objective
                objective = cp.Minimize(L)

                # formulate constraints
                constraints = []
                cons = [-r@y - Y_r@u + (t+h)@s - L]
                for idx in range(n):
                    cons += [y[idx]+Y[idx]@u-s[idx]]
                    cons += [y[idx]+Y[idx]@u-u[idx]]
                constraints += [lropt.max_of_uncertain(cons)<=0]
                constraints += [r@Y == Y_r]
                constraints += [np.ones(n)@s == C]
                constraints += [s <=c, s >=0]
                # formulate Robust Problem
                prob_context = lropt.RobustProblem(objective, constraints)
                prob_context.solve()
                eval, prob_vio = calc_eval(data[test_inds[j]],r,y.value,Y.value,t,h,d,s.value,L.value)
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
                L = cp.Variable()
                s = cp.Variable(n)
                y = cp.Variable(n)
                Y = cp.Variable((n,n))
                r = y_data[j]        
                Y_r = cp.Variable(n)
                # formulate objective
                objective = cp.Minimize(L)

                # formulate constraints
                constraints = []
                cons = [-r@y - Y_r@u + (t+h)@s - L]
                for idx in range(n):
                    cons += [y[idx]+Y[idx]@u-s[idx]]
                    cons += [y[idx]+Y[idx]@u-u[idx]]
                constraints += [lropt.max_of_uncertain(cons)<=0]
                constraints += [r@Y == Y_r]
                constraints += [np.ones(n)@s == C]
                constraints += [s <=c, s >=0]
                # formulate Robust Problem
                prob_context = lropt.RobustProblem(objective, constraints)
                prob_context.solve()
                eval, prob_vio = calc_eval(data[test_inds[j]],r,y.value,Y.value,t,h,d,s.value,L.value)
                nonrob_evals += eval
                nonrob_probs += prob_vio
            nonrob_evals = nonrob_evals / (num_context)
            nonrob_probs = nonrob_probs / (num_context)
            data_df = {'seed': initseed+10*seed, "a_seed":finseed,"nonrob_prob": nonrob_probs, "nonrob_obj":nonrob_evals, "scenario_probs": context_probs, "scenario_obj": context_evals}
            single_row_df = pd.DataFrame(data_df, index=[0])
            single_row_df.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals_nonrob.csv",index=False)

        u = lropt.UncertainParameter(n,
                                        uncertainty_set = lropt.Ellipsoidal(p=2, data =data))
        # formulate cvxpy variable
        L = cp.Variable()
        s = cp.Variable(n)
        y = cp.Variable(n)
        Y = cp.Variable((n,n))
        r = lropt.ContextParameter(n, data = y_data) 
        context = lropt.ContextParameter((n,m), data=context_dat)           
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
        prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp )

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
        settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=True, lr=0.001,epochs = 100)
        try: 
            result = trainer.train(settings=settings)
            df = result.df
            A_fin = result.A
            b_fin = result.b
            torch.save(result._predictor.state_dict(),hydra_out_dir+'/'+str(seed)+'_trained_linear.pth')

            # result_grid4 = trainer.grid(rholst=eps_list,init_A=A_fin, init_b=b_fin, seed=5,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, predictor = result._predictor)
            # dfgrid4 = result_grid4.df
            # dfgrid4 = dfgrid4.drop(columns=["z_vals","x_vals"])
            # dfgrid4.to_csv(hydra_out_dir+'/'+str(seed)+'_linear_trained_grid.csv')
        except:
            print("training failed ",finseed,cfg.eta,cfg.obj_scale)

        try:
            findfs = []
            for rho in eps_list:
                df_valid, df_test = trainer.compare_predictors(settings=settings,predictors_list = [result._predictor], rho_list=[rho])
                data_df = {'seed': initseed+10*seed, 'rho':rho, "a_seed":finseed, 'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'valid_obj': df_valid["Validate_val"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_val"][0], 'test_prob': df_test["Avg_prob_test"][0]}
                single_row_df = pd.DataFrame(data_df, index=[0])
                findfs.append(single_row_df)
                tempdfs = pd.concat(findfs)
                tempdfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
            findfs = pd.concat(findfs)
            findfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
        except:
            print("compare failed")

        if cfg.eta == 0.10 and cfg.obj_scale==0.5:
            settings.init_rho = cfg.init_rho
            settings.num_iter = 1
            settings.initialize_predictor = True
            result_grid = trainer.grid(rholst=eps_list, init_A=init,
                                init_b=init_bval, seed=5,
                                init_alpha=0., test_percentage=test_p, quantiles = (0.3, 0.7))
            dfgrid = result_grid.df
            dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
            dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'mean_var_grid.csv')

            # untrained linear
            settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=True, lr=0.001,epochs = 100)
            settings.num_iter = 1
            result2 = trainer.train(settings=settings)
            A_fin2 = result2.A
            b_fin2 = result2.b
            result_grid3 = trainer.grid(rholst=eps_list,init_A=A_fin2, init_b=b_fin2, seed=5,init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7), contextual = True, predictor = result2._predictor)
            dfgrid3 = result_grid3.df
            dfgrid3 = dfgrid3.drop(columns=["z_vals","x_vals"])
            dfgrid3.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'linear_untrained_grid.csv')
            torch.save(result2._predictor.state_dict(),hydra_out_dir+'/'+str(seed)+'_'+'pretrained_linear.pth')

        try:
            plt.plot(result.df["Train_val"])
            plt.savefig(hydra_out_dir+'/'+str(seed)+'_'+"iters.pdf")
        except:
            None

        beg1, end1 = 0, 100
        beg2, end2 = 0, 100
        plt.figure(figsize=(15, 4))
        
        if cfg.eta == 0.10 and cfg.obj_scale==0.5:
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
            
        # plt.scatter(df_valid["Avg_prob_validate"][0],df_valid["Validate_val"][0], color="tab:orange", label="Linear trained validate set", marker="^", zorder=1)
        # plt.scatter(df_test["Avg_prob_test"][0],df_test["Test_val"][0], color="tab:orange", label="Linear trained test set", marker="s", zorder=1)

        # try:
        #     plt.plot(np.mean(np.vstack(dfgrid4['Avg_prob_test']), axis=1)[beg2:end2], np.mean(np.vstack(
        #         dfgrid4['Test_val']), axis=1)[beg2:end2], color="tab:red", label="Linear trained grid test", marker="s", zorder=2)
        #     plt.plot(np.mean(np.vstack(dfgrid4['Avg_prob_validate']), axis=1)[beg2:end2], np.mean(np.vstack(
        #         dfgrid4['Validate_val']), axis=1)[beg2:end2], color="tab:red", label="Linear trained grid validate", marker="^", zorder=2)
        # except:
        #     None
        plt.ylabel("Objective value")
        plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
        # plt.ylim([-9, 0])
        plt.grid()
        plt.legend()
        plt.savefig(hydra_out_dir+'/'+str(seed)+'_'+"port_objective_vs_violations_"+str(cfg.eta)+".pdf", bbox_inches='tight')

        plt.figure(figsize=(15, 4))
        return None
    except:
        return None

@hydra.main(config_path="/scratch/gpfs/iywang/lropt_revision/lropt_experiments/lropt_experiments/inventory_parallel/configs",config_name = "inv.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # print(f"Current working directory: {os.getcwd()}")
    njobs = get_n_processes(30)
    Parallel(n_jobs=njobs)(
        delayed(inv_exp)(cfg,hydra_out_dir,r) for r in range(R))
    # for r in range(R):
    #     portfolio_exp(cfg,hydra_out_dir,r)
    

if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--foldername', type=str,
    #                     default="portfolio/", metavar='N')
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--R', type=int, default=2)
    # parser.add_argument('--n', type=int, default=15)
    # arguments = parser.parse_args()
    seed_list = [0,50,0,50,0,50]
    m_list= [4,4,8,8,8,8]
    n_list = [10,10,10,10,10,10]
    N_list = [1000,1000,1000,1000,500,500]
    # contxtual = [T,T,F,T,T,T]
    R = 5
    initseed = seed_list[idx]
    test_p = 0.5
    N = N_list[idx]
    #1000
    n = n_list[idx]
    m = m_list[idx]
    #m_list[idx]
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
    test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
    test_indices = test_valid_indices[:int((test_p)*N)]
    valid_indices = test_valid_indices[int((test_p)*N):]
    train_indices = [i for i in range(N) if i not in test_valid_indices]
    context_inds = {}
    test_inds = {}
    for j in range(num_context):
        context_inds[j]= [i for i in train_indices if j*num_reps <= i <= (j+1)*num_reps]
        test_inds[j] = [i for i in test_valid_indices if j*num_reps <= i <= (j+1)*num_reps]
    eps_list=np.linspace(1, 4, 50)
    main_func()

