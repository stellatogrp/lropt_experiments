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
    ax1.plot(np.arange(len_train),dftrain["Violations_train"][:steps],
             label="In-sample empirical CVaR", linestyle="--")

    ax1.set_xlabel("Iterations")
    ax1.hlines(xmin=0, xmax=dftrain["Violations_train"][:steps].shape[0],
               y=-0.0, linestyles="--", color="black", label="Target threshold: 0")
    ax1.legend()
    len_test = len(dftest["Test_val"])
    ax2.plot(np.arange(len_test), dftest["Test_val"][:steps], label="Objective value")
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


def data_scaled(N, m, scale, seed):
    np.random.seed(seed)
    R = np.vstack([np.random.normal(
        i*0.03*scale, np.sqrt((0.02**2+(i*0.1)**2)), N) for i in range(1, m+1)])
    return (R.transpose())

def data_modes(N, m, scales, seed):
    modes = len(scales)
    d = np.zeros((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,
          :] = data_scaled(weights, m, scales[i], seed)
    return d[0:N, :]

def trainloop(r1,foldername):
    seed = r1
    for N in np.array([100]):
        print(N,r1)
        data_gen = False
        test_p = 0.2
        while not data_gen:
            try: 
                data = data_modes(N,m,[1,2,3],seed = seed)
                train, test = train_test_split(data, test_size=int(
                  data.shape[0]*test_p), random_state=seed)
                init = np.real(sc.linalg.sqrtm(np.cov(train.T)))
            except Exception as e:
                seed += 1
            else: 
                data_gen = True

        newdata = data_modes(20000,m,[1,2,3],seed = 10000+seed)
        num_reps = int(N/10)
        y_data1 = np.vstack([y_data]*num_reps)
        num_reps2 = int(20000/10)
        new_y_data = np.vstack([y_data]*num_reps2)
        init_bval = np.mean(train, axis=0)
                
        # formulate the ellipsoidal set
        u = lropt.UncertainParameter(m,
                                        uncertainty_set = lropt.Ellipsoidal(p=2, data =data))
        # formulate cvxpy variable
        L = cp.Variable()
        s = cp.Variable(n)
        y = cp.Variable(n)
        Y = cp.Variable((n,m))
        r = lropt.Parameter(n, data = y_data1)        

        # formulate objective
        objective = cp.Minimize(L)

        # formulate constraints
        constraints = [cp.maximum(-r@y - r@Y@u + (t+h)@s - L, y[0]+Y[0]@u -s[0],y[1]+Y[1]@u -s[1],y[2]+Y[2]@u -s[2],y[3]+Y[3]@u -s[3],y[4]+Y[4]@u -s[4],y[5]+Y[5]@u -s[5],y[6]+Y[6]@u -s[6],y[7]+Y[7]@u -s[7], y[8]+Y[8]@u -s[8],y[9]+Y[9]@u -s[9],y[0] - d[0] - (Q[0] - Y[0])@u,y[1] - d[1] - (Q[1] - Y[1])@u,y[2] - d[2] - (Q[2] - Y[2])@u ,y[3] - d[3] - (Q[3] - Y[3])@u,y[4] - d[4] - (Q[4] - Y[4])@u,y[5] - d[5] - (Q[5] - Y[5])@u,y[6] - d[6] - (Q[6] - Y[6])@u,y[7] - d[7] - (Q[7] - Y[7])@u,y[8] - d[8] - (Q[8] - Y[8])@u,y[9] - d[9] - (Q[9] - Y[9])@u ) <= 0]
        constraints += [np.ones(n)@s == C]
        constraints += [s <=c, s >=0]
        eval_exp = -r@y - r@Y@u + (t+h)@s
        # formulate Robust Problem
        prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp )
        # solve
        # seed 1, 
        result = prob.train(lr = 0.0001,num_iter=1000, num_iter_size = 1000, lr_size= 0.001, train_size = True, optimizer = "SGD", seed = seed, init_A = init, init_b = init_bval, init_lam = 2.0, init_mu =2.0, mu_multiplier=1.02, init_alpha = -0.0, test_percentage = test_p, save_history = False, lr_step_size = 50, lr_gamma = 0.5, position = False, random_init = False, num_random_init=6, parallel = True, eta = eta, kappa=0.)
        A_fin = result.A
        b_fin = result.b
        eps_fin = result.eps

        # Grid search epsilon
        result4 = prob.grid(epslst = np.linspace(0.0001, 6, 100), init_A = init, init_b = init_bval, seed = seed, init_alpha = 0., test_percentage =test_p,newdata = (newdata,new_y_data), eta=eta)
        dfgrid = result4.df

        result5 = prob.grid(epslst = np.linspace(0.0001,6, 100), init_A = A_fin, init_b = b_fin, seed = seed, init_alpha = 0., test_percentage = test_p,newdata = (newdata,new_y_data), eta=eta)
        dfgrid2 = result5.df

        plot_coverage_all(dfgrid,dfgrid2,None, foldername + f"inv(N,m,r)_{N,m,r1}", f"inv(N,m,r)_{N,n,r1}", ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)

        plot_iters(result.df, result.df_test, foldername + f"inv(N,m)_{N,m,r1}", steps = 10000,logscale = 1)

        dfgrid.to_csv(foldername + f"gridmv_{N,m,r1}.csv")
        dfgrid2.to_csv(foldername +f"gridre_{N,m,r1}.csv")
        result.df_test.to_csv(foldername +f"trainval_{N,m,r1}.csv")
        result.df.to_csv(foldername +f"train_{N,m,r1}.csv")



if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="inventory/", metavar='N')
    parser.add_argument('--eta', type=float, default=0.3)
    arguments = parser.parse_args()
    foldername = arguments.foldername
    eta = arguments.eta
    R = 20
    n = 10
    m = 8
    np.random.seed(27)
    y_nom = np.random.uniform(2,4,n)
    y_data = y_nom
    num_scenarios = 9
    for scene in range(num_scenarios):
        np.random.seed(scene)
        y_data = np.vstack([y_data,np.maximum(y_nom + np.random.normal(0,0.05,n),0)])
    np.random.seed(27)
    C = 200
    c = np.random.uniform(30,50,n)
    Q = np.random.uniform(-0.2,0.2,(n,m))
    d = np.random.uniform(10,20,n)
    t = np.random.uniform(0.1,0.3,n)
    h = np.random.uniform(0.1,0.3,n)
    njobs = get_n_processes(30)
    print(foldername)
    Parallel(n_jobs=njobs)(
        delayed(trainloop)(r, foldername) for r in range(R))


    val_st = []
    val_re = []
    prob_st = []
    prob_re = []
    nvals = np.array([100])
    for N in nvals:
        dfgrid = pd.read_csv(foldername +f"gridmv_{N,m,0}.csv")
        dfgrid = dfgrid.drop(columns=["step","Probability_violations_test","var_values"])
        dfgrid2 = pd.read_csv(foldername +f"gridre_{N,m,0}.csv")
        dfgrid2 = dfgrid2.drop(columns=["step","Probability_violations_test","var_values"])
        df_test = pd.read_csv(foldername +f"trainval_{N,m,0}.csv")
        df = pd.read_csv(foldername +f"train_{N,m,0}.csv")
        for r in range(1,R):
            newgrid = pd.read_csv(foldername +f"gridmv_{N,m,r}.csv")
            newgrid = newgrid.drop(columns=["step","Probability_violations_test","var_values"])
            dfgrid = dfgrid.add(newgrid.reset_index(), fill_value=0)
            newgrid2 = pd.read_csv(foldername +f"gridre_{N,m,r}.csv")
            newgrid2 = newgrid2.drop(columns=["step","Probability_violations_test","var_values"])
            dfgrid2 = dfgrid2.add(newgrid2.reset_index(), fill_value=0)

        if R > 1:
            dfgrid = dfgrid/R
            dfgrid2 = dfgrid2/R
            dfgrid.to_csv(foldername + f"results/gridmv_{N,m}.csv")
            dfgrid2.to_csv(foldername +f"results/gridre_{N,m}.csv")

            plot_coverage_all(dfgrid,dfgrid2,None, foldername + f"results/inv(N,m,r)_{N,n}", f"inv(N,m,r)_{N,n,r}", ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)
