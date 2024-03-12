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


def plot_iters(dftrain, dftest, title, steps=2000, logscale=True):
    plt.rcParams.update({
        "text.usetex": True,

        "font.size": 22,
        "font.family": "serif"
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
    len_train = len(dftrain["Violations_train"])
    ax1.plot(np.arange(0,len_train,10), dftest["Violations_test"][:steps],
             label="Out-of-sample empirical CVaR")
    ax1.plot(dftrain["Violations_train"][:steps],
             label="In-sample empirical CVaR", linestyle="--")

    ax1.set_xlabel("Iterations")
    ax1.hlines(xmin=0, xmax=dftrain["Violations_train"][:steps].shape[0],
               y=-0.0, linestyles="--", color="black", label="Target threshold: 0")
    ax1.legend()
    ax2.plot(dftest["Test_val"][:steps], label="Objective value")
    ax2.set_xlabel("Iterations")
    ax2.ticklabel_format(style="sci", axis='y',
                         scilimits=(0, 0), useMathText=True)
    ax2.legend()
    if logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    plt.savefig(title+"_iters", bbox_inches='tight')
    plt.show()

def plot_coverage_all(df_standard,df_reshape,dfs,title,title1,ind_1 = (0,100), ind_2 = (0,100), logscale = True, legend = False, zoom = False):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    beg1,end1 = ind_1
    beg2,end2 = ind_2

    fig, (ax, ax1,ax2) = plt.subplots(1, 3, figsize=(23, 3))
    
    ax.plot(np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], df_standard['Test_val'][beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax.fill(np.append(np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1],np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1][::-1]), np.append(df_standard['Lower_test'][beg1:end1],df_standard['Upper_test'][beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax.plot(np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], df_reshape['Test_val'][beg2:end2], color="tab:orange", label=r"Reshaped set")
    ax.fill(np.append(np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2][::-1]), np.append(df_reshape['Lower_test'][beg2:end2],df_reshape['Upper_test'][beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax.set_xlabel("Probability of constraint violation")
    ax.axvline(x = 0.03, color = "green", linestyle = "-.",label = r"$\eta = 0.03$")

    
    ax.set_ylabel("Objective value")
    ax.set_title(title1)
    ax.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)

    ax1.plot(np.mean(np.vstack(df_standard['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax1.fill(np.append(np.quantile(np.vstack(df_standard['Coverage_test']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Coverage_test']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.90,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax1.plot(np.mean(np.vstack(df_reshape['Coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",label=r"Decision-Focused set")
    ax1.fill(np.append(np.quantile(np.vstack(df_reshape['Coverage_test']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Coverage_test']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.90,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax1.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax1.axvline(x = 0.8, color = "black", linestyle = ":",label = "0.8 Coverage")

    if logscale:
        ax1.set_xscale("log")
    # ax1.set_yticks(ticks = [-1,0,1])

    ax1.set_xlabel("Test set coverage")
    ax1.set_ylabel("Objective value")
    # ax1.legend()

    ax2.plot(df_standard['Coverage_test'][beg1:end1], np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")

    ax2.plot(df_reshape['Coverage_test'][beg2:end2], np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], color="tab:orange", label=r"Reshaped set",alpha = 0.8)
    ax2.axvline(x = 0.8, color = "black",linestyle = ":", label = "0.8 Coverage")
    ax2.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
    ax2.set_ylabel("Prob. of cons. vio.")
    ax2.set_xlabel("Test set coverage")
    if zoom:
        axins = zoomed_inset_axes(ax2, 6, loc="upper center")
        axins.set_xlim(-0.005, 0.1)
        axins.set_ylim(-0.001,0.035)
        axins.plot(np.mean(np.vstack(df_standard['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], color="tab:blue")
        axins.plot(np.mean(np.vstack(df_reshape['Coverage_test']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], color="tab:orange",alpha = 0.8)
        axins.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
        axins.set_xticks(ticks=[])
        axins.set_yticks(ticks=[])
        mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    if logscale:
        ax2.set_xscale("log")
    if legend:
        ax2.legend(bbox_to_anchor=(-1.8, -0.6, 0, 0), loc="lower left",
                 borderaxespad=0, ncol=4, fontsize = 24)
    plt.subplots_adjust(left=0.1)
    plt.savefig(title+"_curves",bbox_inches='tight')
    plt.show()




def gen_sigmu(n,seed = 0):
    np.random.seed(seed)
    F = np.random.normal(size = (n,2))
    sig = 0.1*F@(F.T)
    mu = np.random.uniform(0.5,1,n)
    return sig, mu

def gen_demand(sig,mu,N,seed=399):
    np.random.seed(seed)
    d_train = np.random.multivariate_normal(mu,sig, N)
    return d_train


def trainloop(r,foldername):
    seed = r + 10
    for N in np.array([500]):
        print(N,r)
        data_gen = False
        test_p = 0.2
        while not data_gen:
            try: 
                data = gen_demand(sig,mu,N,seed=seed)
                train, test = train_test_split(data, test_size=int(
                  data.shape[0]*test_p), random_state=seed)
                init = sc.linalg.sqrtm(np.cov(train.T)+0.001*np.eye(n))
            except Exception as e:
                seed += 1
            else: 
                data_gen = True
        newdata = gen_demand(sig,mu,20000,seed=10000+seed)
        y_data = np.maximum(y_nom + np.random.normal(0,0.05,(10,n)),0.001)
        y_data = np.diag(1/np.sum(y_data, axis=1))@y_data
        num_reps = int(N/10)
        y_data = np.vstack([y_data]*num_reps)

        new_y_data = np.maximum(y_nom + np.random.normal(0,0.05,(10,n)),0.001)
        new_y_data = np.diag(1/np.sum(new_y_data, axis=1))@new_y_data
        num_reps2 = int(20000/10)
        new_y_data = np.vstack([new_y_data]*num_reps2)

        init_bval = np.mean(train, axis=0)
                
        u = lropt.UncertainParameter(n,
                                uncertainty_set=lropt.Ellipsoidal(p=2,
                                                            data=data))
        # Formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()
        y = lropt.Parameter(n, data=y_data)

        objective = cp.Minimize(t + 0.2*cp.norm(x - y, 1))
        constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]
        eval_exp = -x @ u + 0.2*cp.norm(x-y, 1)

        prob = lropt.RobustProblem(objective, constraints, eval_exp=eval_exp)
        s = seed

        # Train A and b
        result = prob.train(lr=0.01, num_iter=3000, optimizer="SGD",
                            seed=s, init_A=0.5*init, init_b=init_bval, init_lam=1, init_mu=1,
                            mu_multiplier=1.005, init_alpha=0., test_percentage = test_p, save_history = False, lr_step_size = 300, lr_gamma = 0.2, position = False, random_init = True, num_random_init=5, parallel = True, eta = eta, kappa=0.0)
        df = result.df
        A_fin = result.A
        b_fin = result.b
        epslst=np.linspace(0.00001, 10, 200)
        result5 = prob.grid(epslst=epslst, init_A=result.eps*A_fin, init_b=b_fin, seed=s,
                            init_alpha=0., test_percentage=test_p, newdata = (newdata,new_y_data), eta=eta)
        dfgrid2 = result5.df
        result4 = prob.grid(epslst=epslst, init_A=init,
                            init_b=init_bval, seed=s,
                            init_alpha=0., test_percentage=test_p, newdata=(newdata,new_y_data), eta=eta)
        dfgrid = result4.df

        plot_coverage_all(dfgrid,dfgrid2,None, foldername + f"port(N,m,r)_{N,n,r}", f"port(N,m,r)_{N,n,r}", ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)

        plot_iters(df, result.df_test, foldername + f"port(N,m)_{N,n,r}", steps = 10000,logscale = 1)

        dfgrid.to_csv(foldername + f"gridmv_{N,n,r}.csv")
        dfgrid2.to_csv(foldername +f"gridre_{N,n,r}.csv")
        result.df_test.to_csv(foldername +f"trainval_{N,n,r}.csv")
        result.df.to_csv(foldername +f"train_{N,n,r}.csv")



if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="portfolio/", metavar='N')
    parser.add_argument('--eta', type=float, default=0.05)
    arguments = parser.parse_args()
    foldername = arguments.foldername
    eta = arguments.eta
    R = 20
    n = 5
    seed = 25
    sig, mu = gen_sigmu(n,1)
    np.random.seed(seed)
    dist = mu
    y_nom = np.random.dirichlet(dist)
    njobs = get_n_processes(30)
    print(foldername)
    Parallel(n_jobs=njobs)(
        delayed(trainloop)(r, foldername) for r in range(R))
    
    val_st = []
    val_re = []
    prob_st = []
    prob_re = []
    nvals = np.array([500])
    for N in nvals:
        dfgrid = pd.read_csv(foldername +f"gridmv_{N,n,0}.csv")
        dfgrid = dfgrid.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
        dfgrid2 = pd.read_csv(foldername +f"gridre_{N,n,0}.csv")
        dfgrid2 = dfgrid2.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
        df_test = pd.read_csv(foldername +f"trainval_{N,n,0}.csv")
        df = pd.read_csv(foldername +f"train_{N,n,0}.csv")
        for r in range(1,R):
            newgrid = pd.read_csv(foldername +f"gridmv_{N,n,r}.csv")
            newgrid = newgrid.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
            dfgrid = dfgrid.add(newgrid.reset_index(), fill_value=0)
            newgrid2 = pd.read_csv(foldername +f"gridre_{N,n,r}.csv")
            newgrid2 = newgrid2.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
            dfgrid2 = dfgrid2.add(newgrid2.reset_index(), fill_value=0)
