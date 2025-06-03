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
import matplotlib.pyplot as plt
import hydra
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})
path = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/port_parallel/plots/plots_0/20/"
R = 5
etas = [0.01,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.23,0.25,0.30,0.33,0.35,0.40]
objs = [0.25,0.5,1]
seeds1 = [0,10,20,30,40]
seeds2 = [50,60,70,80,90]
foldername1 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/2025-04-28/20_s0/"
foldername2 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/2025-04-28/20_s1/"
dfs_all = {}
quantiles = [0.25,0.75]
dfs = {}
running_ind = 0
for obj in objs:
    dfs_all[obj] = []
for eta in etas:
    for obj in objs:
        newfolder = foldername1+str(running_ind)
        for seed in range(R):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_all[obj].append(df)
            except:
                print(1,eta,obj,seed)
        running_ind += 1
running_ind = 0
for eta in etas:
    for obj in objs:
        newfolder = foldername2+str(running_ind)
        for seed in range(R):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_all[obj].append(df)
            except:
                print(1,eta,obj,seed)
        running_ind += 1
for obj in objs:
    dfs_all[obj] = pd.concat(dfs_all[obj])
collist = list(dfs_all[obj].columns)[5:]
ecollist = ["eta"]+ collist
for obj in objs:
    grouped = dfs_all[obj].groupby(["eta"], as_index=False)
    mean_vals = grouped[collist].mean().add_prefix("mean_")
    dfs[obj] = mean_vals
    for q in quantiles:
        quantile_values = grouped[collist].quantile(q)
        quantile_values = grouped[collist].quantile(q).add_prefix(str(q)+"_")
        dfs[obj] = pd.concat([dfs[obj], quantile_values], axis=1)
    dfs[obj].to_csv(path+"gamma_"+str(obj)+"_values.csv")
df_pre = []
running_ind = 21
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_linear_untrained_grid.csv")
        df['seed'] = seed
        df_pre.append(df)
    except:
        print(2,eta,obj,seed)
newfolder = foldername2+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_linear_untrained_grid.csv")
        df['seed'] = seed
        df_pre.append(df)
    except:
        print(2,eta,obj,seed)
df_pre = pd.concat(df_pre)
dfs_pre = []
collist_grid = ["Test_val","Avg_prob_test","Validate_val","Avg_prob_validate"]
grouped = df_pre.groupby(["Rho"], as_index=False)
mean_vals = grouped[collist_grid].mean().add_prefix("mean_")
dfs_grid = mean_vals
for q in quantiles:
    quantile_values = grouped[collist_grid].quantile(q)
    quantile_values = grouped[collist_grid].quantile(q).add_prefix(str(q)+"_")
    dfs_grid = pd.concat([dfs_grid, quantile_values], axis=1)
dfs_grid.to_csv(path+"pretrained.csv")
df_mv = []
running_ind = 21
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_mean_var_grid.csv")
        df['seed'] = seed
        df_mv.append(df)
    except:
        print(3,eta,obj,seed)
newfolder = foldername2+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_mean_var_grid.csv")
        df['seed'] = seed
        df_mv.append(df)
    except:
        print(3,eta,obj,seed)
df_mv = pd.concat(df_mv)
dfs_mv = []
collist_grid = ["Test_val","Avg_prob_test","Validate_val","Avg_prob_validate"]
grouped = df_mv.groupby(["Rho"], as_index=False)
mean_vals = grouped[collist_grid].mean().add_prefix("mean_")
dfs_mv_grid = mean_vals
for q in quantiles:
    quantile_values = grouped[collist_grid].quantile(q)
    quantile_values = grouped[collist_grid].quantile(q).add_prefix(str(q)+"_")
    dfs_mv_grid = pd.concat([dfs_mv_grid, quantile_values], axis=1)
dfs_mv_grid.to_csv(path+"pretrained.csv")

# plt.rcParams.update({
#     "text.usetex":True,
#     "font.size":24,
#     "font.family": "sans-serif"
# })
def plot_compare(dfs,idx,dfs_grid,dfs_mv_grid,valid,ylim=None):
    plt.figure(figsize = (8,4))
    if valid: 

        plt.plot(np.array(dfs_mv_grid["mean_Avg_prob_validate"]),np.array(dfs_mv_grid["mean_Validate_val"]),label = "Mean-var validation")
        plt.fill_between(np.array(dfs_mv_grid["0.25_Avg_prob_validate"]),np.array(dfs_mv_grid["0.25_Validate_val"]),np.array(dfs_mv_grid["0.75_Validate_val"]),alpha = 0.25)

        plt.plot(np.array(dfs[idx]["mean_valid_prob"]),np.array(dfs[idx]["mean_valid_obj"]),label = "Trained validation")
        plt.fill_between(np.array(dfs[idx]["mean_valid_prob"]),np.array(dfs[idx]["0.25_valid_obj"]),np.array(dfs[idx]["0.75_valid_obj"]),alpha = 0.25)

        plt.plot(np.array(dfs_grid["mean_Avg_prob_validate"]),np.array(dfs_grid["mean_Validate_val"]),label = "Pre-trained validation")
        plt.fill_between(np.array(dfs_grid["0.25_Avg_prob_validate"]),np.array(dfs_grid["0.25_Validate_val"]),np.array(dfs_grid["0.75_Validate_val"]),alpha = 0.25)
    else:
        plt.plot(np.array(dfs_mv_grid["mean_Avg_prob_test"]),np.array(dfs_mv_grid["mean_Test_val"]),label = "Mean-var test")
        plt.fill_between(np.array(dfs_mv_grid["0.25_Avg_prob_test"]),np.array(dfs_mv_grid["0.25_Test_val"]),np.array(dfs_mv_grid["0.75_Test_val"]),alpha = 0.25)
        
        plt.plot(np.array(dfs[idx]["mean_test_prob"]),np.array(dfs[idx]["mean_test_obj"]),label = "Trained test")
        plt.fill_between(np.array(dfs[idx]["mean_test_prob"]),np.array(dfs[idx]["0.25_test_obj"]),np.array(dfs[idx]["0.75_test_obj"]),alpha = 0.25)
        
        plt.plot(np.array(dfs_grid["mean_Avg_prob_test"]),np.array(dfs_grid["mean_Test_val"]),label = "Pre-trained test")
        plt.fill_between(np.array(dfs_grid["0.25_Avg_prob_validate"]),np.array(dfs_grid["0.25_Test_val"]),np.array(dfs_grid["0.75_Test_val"]),alpha = 0.25)
        
    plt.plot(np.array(dfs[idx]["mean_nonrob_prob"]),np.array(dfs[idx]["mean_nonrob_obj"]),label = "Non-robust")
    plt.fill_between(np.array(dfs[idx]["mean_nonrob_prob"]),np.array(dfs[idx]["0.25_nonrob_obj"]),np.array(dfs[idx]["0.75_nonrob_obj"]),alpha = 0.25)
    plt.plot(np.array(dfs[idx]["mean_scenario_probs"]),np.array(dfs[idx]["mean_scenario_obj"]),label = "Scenario")
    plt.fill_between(np.array(dfs[idx]["mean_scenario_probs"]),np.array(dfs[idx]["0.25_scenario_obj"]),np.array(dfs[idx]["0.75_scenario_obj"]),alpha = 0.25)
    plt.legend()
    plt.xlabel("Prob of constr. violation")
    plt.ylabel("Out-of-sample objective")
    plt.ylim(ylim)
    plt.tight_layout()
    if valid:
        plt.title("Validation objectives_"+"gamma_"+str(idx))
        plt.savefig(path+"Validation objectives_"+"gamma_"+str(idx)+".pdf")
    else:
        plt.title("Test objectives_"+"gamma_"+str(idx))
        plt.savefig(path+"Test objectives_"+"gamma_"+str(idx)+".pdf")
for idx in objs:
    plot_compare(dfs,idx,dfs_grid,dfs_mv_grid,valid=True)
    plot_compare(dfs,idx,dfs_grid,dfs_mv_grid,valid=False)

dfs_cat = []
running_ind = 0
for eta in etas:
    for obj in objs:
        newfolder = foldername1+str(running_ind)
        for seed in range(5):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_cat.append(df)
            except:
                print(4,eta,obj,seed)
        running_ind += 1
running_ind = 0
for eta in etas:
    for obj in objs:
        newfolder = foldername2+str(running_ind)
        for seed in range(5):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_cat.append(df)
            except:
                print(4,eta,obj,seed)
        running_ind += 1
dfs_cat = pd.concat(dfs_cat)
inds = {}
#target_list = [0.01,0.05,0.1,0.15,0.20]
target_list = [0.01,0.02,0.03,0.05,0.08,0.1,0.12,0.15,0.18,0.20]
dfs_best = {}
dif=0.01
for target in target_list:
    inds[target] = []
    dfs_best[target] = []
    for seed in seeds1+seeds2:
        try:
            best_idx = np.argmin(dfs_cat[dfs_cat["seed"] == seed][abs(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)<=dif]["valid_obj"])
            inds[target].append(best_idx)
            cur_df = dfs_cat[dfs_cat["seed"] == seed][abs(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)<=dif].iloc[best_idx:best_idx+1]
            dfs_best[target].append(cur_df)
            # best_idx = np.argmin(np.abs(np.array(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)))
            # inds[target].append(best_idx)
            # cur_df = dfs_cat[dfs_cat["seed"] == seed].iloc[best_idx:best_idx+1]
            # dfs_best[target].append(cur_df)
        except:
            print(seed)
    dfs_best[target] = pd.concat(dfs_best[target])   
plot_data = []
for target in target_list:
    data = {'test_prob': dfs_best[target]["test_prob"].mean(),'test_obj': dfs_best[target]["test_obj"].mean(),'0.25_test_obj': dfs_best[target]["test_obj"].quantile(0.25), '0.75_test_obj': dfs_best[target]["test_obj"].quantile(0.75)}
    data = pd.DataFrame(data, index=[0])
    plot_data.append(data)
plot_data = pd.concat(plot_data)
plot_data.to_csv(path+"plot_data.csv") 
dfs_best[0.05].to_csv(path+"plot_data_005.csv") 
dfs_best[0.1].to_csv(path+"plot_data_01.csv") 
dfs_best[0.15].to_csv(path+"plot_data_015.csv")
# plt.rcParams.update({
#     "text.usetex":True,
#     "font.size":24,
#     "font.family": "sans-serif"
# })
def plot_best(plot_data,dfs,dfs_grid,dfs_mv_grid,ylim=None):
    idx = 1
    plt.figure(figsize = (8,4))
    plt.plot(np.array(dfs_mv_grid["mean_Avg_prob_test"]),np.array(dfs_mv_grid["mean_Test_val"]),label = "Mean-var test",marker = "^")
    plt.fill_between(np.array(dfs_mv_grid["0.25_Avg_prob_test"]),np.array(dfs_mv_grid["0.25_Test_val"]),np.array(dfs_mv_grid["0.75_Test_val"]),alpha = 0.25)
    
    plt.plot(np.array(plot_data["test_prob"]),np.array(plot_data["test_obj"]), label = "Trained test" ,color = "tab:orange",marker = "D")
    plt.fill_between(np.array(plot_data["test_prob"]),np.array(plot_data["0.25_test_obj"]),np.array(plot_data["0.75_test_obj"]),alpha = 0.25, color = "tab:orange")

    plt.plot(np.array(dfs_grid["mean_Avg_prob_test"]),np.array(dfs_grid["mean_Test_val"]),label = "Pre-trained test",marker = "v",color = "tab:green")
    plt.fill_between(np.array(dfs_grid["0.25_Avg_prob_validate"]),np.array(dfs_grid["0.25_Test_val"]),np.array(dfs_grid["0.75_Test_val"]),alpha = 0.25,color = "tab:green")
        
    plt.plot(np.array(dfs[idx]["mean_nonrob_prob"])[1:],np.array(dfs[idx]["mean_nonrob_obj"])[1:],label = "Non-robust",color = "tab:pink",marker = "s")
    plt.fill_between(np.array(dfs[idx]["mean_nonrob_prob"])[1:],np.array(dfs[idx]["0.25_nonrob_obj"])[1:],np.array(dfs[idx]["0.75_nonrob_obj"])[1:],alpha = 0.25,color = "tab:pink")
    plt.plot(np.array(dfs[idx]["mean_scenario_probs"])[1:],np.array(dfs[idx]["mean_scenario_obj"])[1:],label = "Scenario",color = "tab:purple",marker = "o")
    plt.fill_between(np.array(dfs[idx]["mean_scenario_probs"])[1:],np.array(dfs[idx]["0.25_scenario_obj"])[1:],np.array(dfs[idx]["0.75_scenario_obj"])[1:],alpha = 0.25,color = "tab:purple")
    plt.vlines(target_list,ymin = -0.92,ymax=-0.8,linestyles=":",color = "red",alpha = 0.5)
    # plt.vlines([0.03,0.05,0.10],ymin = -6,ymax=-2,linestyles=":",color = "red",alpha = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("Probability of constraint violation")
    # plt.ylabel("Out-of-sample objective")
    plt.ylim(ylim)
    plt.tight_layout()
    # plt.xlim([-0.02,0.20])
    plt.title("Out-of-sample objectives (test set)")
    plt.savefig(path+"Test_objectives_best_all.pdf",bbox_inches='tight')
plot_best(plot_data,dfs,dfs_grid,dfs_mv_grid)
