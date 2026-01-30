import os
import sys
import joblib
from joblib import Parallel, delayed
from torch import tensor
output_stream = sys.stdout
import cvxpy as cp
import scipy as sc
import numpy as np
import pandas as pd
import lropt
import hydra
import matplotlib.pyplot as plt
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

def gen_demand_varied(d, cov, N, cval=None, pval=None, hval=None, seed=399):
    """Generate demand data, optionally correlated to cval, pval, hval"""

    pointlist = []
    np.random.seed(seed)
    for i in range(N):
        d_train = np.random.multivariate_normal(d - cval[i],cov)
        pointlist.append(d_train)
    return np.vstack(pointlist)


def inv_exp(cfg,hydra_out_dir,seed):
    finseed = initseed + 10*seed
    print(finseed)
    try: 
        data_gen = False
        while not data_gen:
            try: 
                data = gen_demand_varied(d_star, cov, N, cval=cval, pval=pval, hval=hval, seed=finseed)
                train = data[train_indices]
                init = sc.linalg.sqrtm(np.cov(train.T)+0.00001*np.eye(T))
                init_bval = np.mean(train, axis=0)
            except Exception as e:
                finseed += 1
            else: 
                data_gen = True


        d = lropt.UncertainParameter(T,uncertainty_set = lropt.Ellipsoidal(p=2,rho=1,c = lhs, d = rhs,data = data))
        # d = d_star
        q = cp.Variable(T)
        # y = cp.Variable(T)
        # u = cp.Variable(T)
        # z = cp.Variable(T-1)
        # w = cp.Variable(T)
        # u_var = cp.Variable((T,T))
        # y_var = cp.Variable((T,T))
        q_var = cp.Variable((T,T))
        C = cp.Variable()
        x_init = init_val
        
        # Make cval, hval and pval parameters
        c_param = lropt.ContextParameter(T, data = cval)
        # h_param = lropt.ContextParameter(T, data = hval)
        # p_param = lropt.ContextParameter(T, data = pval)
        
        # Auxiliary variables to replace h_param and p_param where they multiply uncertain parameters
        # h_aux = cp.Variable(T)
        # p_aux = cp.Variable(T)
        
        # Auxiliary variables for h_aux[time] * sum((q_var@d)[:time+1])
        # Define hqvd[time] to represent h_aux[time] * sum((q_var@d)[:time+1])
        # hqvd = cp.Variable((T, T)) 
        # pqvd = cp.Variable((T, T))  
        cqv = cp.Variable(T)  # cqv[t3] represents c_param @ q_var

        objective = C
        # constraints = [c_param@q + cqv@d + cp.sum(y + y_var@d) + cp.sum(u + u_var@d) + cp.sum(z) - C <=0]
        cons = [c_param@q + cqv@d - C]
        constraints = []
        
        # Link auxiliary variables to parameters
        # for time in range(T):
        #     constraints += [h_aux[time] == h_param[time]]
        #     constraints += [p_aux[time] == p_param[time]]
        #     # Link hqvd and pqvd: they represent h_aux[time] * sum of q_var rows
        #     # hqvd[time, :] = h_aux[time] * sum_{t2=0}^{time} q_var[t2, :]
        #     constraints += [hqvd[time, :] == h_param[time] * cp.sum(q_var[:(time+1), :], axis=0)]
        #     constraints += [pqvd[time, :] == p_param[time] * cp.sum(q_var[:(time+1), :], axis=0)]
        
        # Link cqv: cqv = c_param @ q_var
        constraints += [cqv == c_param @ q_var]
        
        # constraints += [y_var == 0, u_var==0]
        for time in range(T):
            for time2 in range(time,T):
                constraints += [q_var[time,time2] == 0]
                # constraints += [y_var[time,time2] == 0]
                # constraints += [u_var[time,time2] == 0]

        for time in range(T):
            cons +=[Vmin -x_init - cp.sum((q+q_var@d)[:time+1]) + cp.sum(d[:(time+1)])]
            cons+=[x_init + cp.sum((q+q_var@d)[:(time+1)]) - cp.sum(d[:(time+1)])- Vmax]
            # constraints += [-(y + y_var@d)[time] + h_param[time]*x_init + h_param[time]*cp.sum(q[:(time+1)]) + hqvd[time, :]@d - h_aux[time]*cp.sum(d[:(time+1)]) <=0]
            # constraints += [-(y + y_var@d)[time] -p_param[time]*x_init - p_param[time]*cp.sum(q[:(time+1)]) - pqvd[time, :]@d + p_aux[time]*cp.sum(d[:(time+1)]) <=0]
            # constraints += [(alpha1*(q+q_var@d -w)-u-u_var@d)[time] <=0]
            # constraints += [(alpha2*(w - q - q_var@d)- u - u_var@d)[time] <=0]
            cons += [-(q+q_var@d)[time]]
            cons += [(q+q_var@d - Qmax)[time]]
        cons += [cp.sum(q+q_var@d)-Pmax]
        # constraints += [alpha1*(q+q_var@d -w) <= u + u_var@d ]
        # constraints += [alpha2*(w-q-q_var@d)<=  u + u_var@d ]
        # constraints += [beta1*(w[1:] - w[:-1]) <= z]
        # constraints += [beta2*(w[:-1] - w[1:]) <= z]
        constraints += [lropt.max_of_uncertain(cons)<=0]
        prob = lropt.RobustProblem(cp.Minimize(objective), constraints, eval_exp = c_param@q + cqv@d)

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
        settings.predictor = lropt.LinearPredictor(predict_mean = True,predict_cov = False, pretrain=True, lr=0.001,epochs = 100,n_neighbors=int(N*0.1*0.3),knn_cov = True, knn_scale = cfg.knn_mult)
        # settings.predictor = lropt.DeepNormalModel()
        settings.data=data
        settings.target_eta = cfg.target_eta
        try: 
            result = trainer.train(settings=settings)
        except:
            print("training failed ",finseed,cfg.eta,cfg.obj_scale)
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
                data_df = {'seed': initseed+10*seed, 'rho':rho, "a_seed":finseed, 'eta':cfg.eta, 'gamma': cfg.obj_scale, 'init_rho': cfg.init_rho, 'valid_obj': df_valid["Validate_worst"][0], 'valid_prob': df_valid["Avg_prob_validate"][0],'test_obj': df_test["Test_worst"][0], 'test_prob': df_test["Avg_prob_test"][0],"time": solvetime,"valid_cover":df_valid["Coverage_validate"][0], "test_cover": df_test["Coverage_test"][0], "valid_in": df_valid["Validate_insample"][0], "test_in": df_test["Test_insample"][0], "avg_val": df_test["Test_val"][0],'valid_cvar': df_valid["Validate_cvar"][0], 'test_cvar': df_test["Test_cvar"][0]}
                single_row_df = pd.DataFrame(data_df, index=[0])
                findfs.append(single_row_df)
                tempdfs = pd.concat(findfs)
                tempdfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
            findfs = pd.concat(findfs)
            findfs.to_csv(hydra_out_dir+'/'+str(seed)+'_'+"vals.csv",index=False)
        except:
            print("compare failed")

        if cfg.eta == 0.05 and cfg.obj_scale==0.5:
            settings.init_rho = cfg.init_rho
            settings.num_iter = 0
            settings.contextual = False
            result_grid = trainer.grid(rholst=eps_list,settings=settings)
            dfgrid = result_grid.df
            dfgrid = dfgrid.drop(columns=["z_vals","x_vals"])
            dfgrid.to_csv(hydra_out_dir+'/'+str(seed)+'_'+'mean_var_grid.csv')

            # untrained linear
            settings.contextual = True
            settings.initialize_predictor = True
            settings.predictor = lropt.LinearPredictor(predict_mean = True,pretrain=False, lr=0.001,epochs = 100,knn_cov=True,n_neighbors=int(N*0.3*0.1),knn_scale = cfg.knn_mult)
            settings.num_iter = 0
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
    except:
        return None

@hydra.main(config_path="configs",config_name = "inv.yaml", version_base = None)
def main_func(cfg):
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    njobs = get_n_processes(30)
    for r in range(R):
        inv_exp(cfg, hydra_out_dir, r)

def plot_demand_comparison(d_star, cval, pval, hval, N, T):
    """Plot original demand vs new demand"""
    # Generate original and new demand
    np.random.seed(0)
    cov = 1000*np.eye(T)
    d_original = np.random.multivariate_normal(d_star, cov, size=N)
    d_new = gen_demand_varied(d_star, cov, N, cval=cval, pval=pval, hval=hval, seed=0)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original demand
    axes[0].boxplot(d_original, labels=[f't{i}' for i in range(T)])
    axes[0].set_title('Original Demand')
    axes[0].set_ylabel('Demand')
    axes[0].grid(True, alpha=0.3)
    
    # Plot new demand
    axes[1].boxplot(d_new, labels=[f't{i}' for i in range(T)])
    axes[1].set_title('New Demand (Correlated to c, p, h)')
    axes[1].set_ylabel('Demand')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demand_comparison.png', dpi=150)
    plt.show()
    print("Saved demand comparison plot to demand_comparison.png")
    

    

if __name__ == "__main__":
    R = 1
    initseed = 0
    test_p = 0.5
    N = 500
    T = 12
    Qmax = 1600
    Pmax = 20000
    Vmin = 100
    Vmax = 2000
    # alpha1 = 0
    # alpha2 = 0
    # beta1 = 1
    # beta2 = 2
    init_val = 500
    cval_base = np.array([(1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)])*0.1
    pval_base = cval_base*2.5
    hval_base = cval_base*1.2
    # Generate random cval, pval, hval with size N using multivariate normal
    np.random.seed(0)
    cov_c = 0.01*np.eye(T)
    cval = np.random.multivariate_normal(cval_base, cov_c, size=N)
    pval = np.random.multivariate_normal(pval_base, cov_c, size=N)
    hval = np.random.multivariate_normal(hval_base, cov_c, size=N)
    d_star = np.array([1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)])
    proportion = 0.1
    lhs = np.concatenate((np.eye(T), -np.eye(T)), axis=0)    
    rhs_upper = (1 + proportion) * d_star
    rhs_lower = (-1 + proportion) * d_star
    rhs = np.hstack((rhs_upper, rhs_lower))
    cov = 500*np.eye(T)

    # np.random.seed(27)
    # y_nom = np.random.uniform(2,4,n)
    # y_data = y_nom
    # num_context = 10
    # num_reps = int(N/num_context)
    # for scene in range(num_context-1):
    #     np.random.seed(scene)
    #     y_data = np.vstack([y_data,np.maximum(y_nom + np.random.normal(0,0.1,n),0)])
    # np.random.seed(27)
    # sig, context = gen_sigmu_varied(n,m,num_context,seed= 0)
    # sig = np.vstack([sig]*num_reps)
    # context_dat = np.vstack([context]*num_reps)
    # y_data = np.vstack([y_data]*num_reps)
    np.random.seed(5)
    test_valid_indices = np.random.choice(N,int((test_p+0.2)*N), replace=False)
    test_indices = test_valid_indices[:int((test_p)*N)]
    valid_indices = test_valid_indices[int((test_p)*N):]
    train_indices = [i for i in range(N) if i not in test_valid_indices]
    # context_inds = {}
    # test_inds = {}
    # for j in range(num_context):
    #     context_inds[j]= [i for i in  train_indices + list([*valid_indices]) if j*num_reps <= i <= (j+1)*num_reps]
    #     test_inds[j] = [i for i in test_indices if j*num_reps <= i <= (j+1)*num_reps]
    eps_list= np.concat([np.logspace(-4,-1,2),np.linspace(0.105,2,5),np.linspace(2.05,3,5),np.linspace(3.1,6,10)])
    
    # Plot demand comparison
    # plot_demand_comparison(d_star, cval, pval, hval, N, T)
    
    main_func()

