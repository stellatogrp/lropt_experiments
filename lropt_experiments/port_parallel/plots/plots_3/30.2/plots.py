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
path = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/port_parallel/plots/plots_3/30.2/"
R = 10
etas = [0.05,0.08,0.09,0.10,0.12,0.15]
objs = [1,1.5,2.5,3,5,8,10]
etas1 = [0.05,0.1,0.12,0.15,0.2,0.3]
objs1 = [0.9,0.6,0.5,0.4,0.2,0.1]
seeds1 = [0,10,20,30,40,50,60,70,80,90]
foldername1 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/cvar/fixed/30_1000/"
foldername4 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/cvar/obj_1/30_1000/"
foldername3 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/cvar/DRO/30_1000/"
foldername5 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/cvar/Delage/30_1000/"
foldername6 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/cvar/lcx/30_1000/"
dfs_all = {}
quantiles = [0.25,0.75]
dfs = {}
running_ind = 0
for obj in objs:
    dfs_all[obj] = []
for eta in etas:
    for obj in objs:
        newfolder = foldername4+str(running_ind)
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
    # dfs[obj].to_csv(path+"gamma_"+str(obj)+"_values.csv")
df_pre = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_linear_pretrained_grid.csv")
        df['seed'] = seeds1[seed]
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
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_mean_var_grid.csv")
        df['seed'] = seeds1[seed]
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
df_dro = []
running_ind = 0
newfolder = foldername3+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_dro_grid.csv")
        df['seed'] = seeds1[seed]
        df_dro.append(df)
    except:
        print(3,eta,obj,seed)
df_dro = pd.concat(df_dro)

df_nonrob = []
running_ind = 1
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_vals_nonrob.csv")
        df_nonrob.append(df)
    except:
        print(4,eta,obj,seed)
df_nonrob = pd.concat(df_nonrob)
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
        
    # plt.plot(np.array(dfs[idx]["mean_nonrob_prob"]),np.array(dfs[idx]["mean_nonrob_obj"]),label = "Non-robust")
    # plt.fill_between(np.array(dfs[idx]["mean_nonrob_prob"]),np.array(dfs[idx]["0.25_nonrob_obj"]),np.array(dfs[idx]["0.75_nonrob_obj"]),alpha = 0.25)
    # plt.plot(np.array(dfs[idx]["mean_scenario_probs"]),np.array(dfs[idx]["mean_scenario_obj"]),label = "Scenario")
    # plt.fill_between(np.array(dfs[idx]["mean_scenario_probs"]),np.array(dfs[idx]["0.25_scenario_obj"]),np.array(dfs[idx]["0.75_scenario_obj"]),alpha = 0.25)
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
# for idx in objs:
#     plot_compare(dfs,idx,dfs_grid,dfs_mv_grid,valid=True)
#     plot_compare(dfs,idx,dfs_grid,dfs_mv_grid,valid=False)

dfs_cat_o = []
running_ind = 0
for eta in etas1:
    for obj in objs1:
        newfolder = foldername5+str(running_ind)
        for seed in range(R):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_cat_o.append(df)
            except:
                print(4,eta,obj,seed)
        running_ind += 1
dfs_cat_o = pd.concat(dfs_cat_o)
dfs_cat = []
running_ind = 0
for eta in etas:
    for obj in objs:
        newfolder = foldername4+str(running_ind)
        for seed in range(R):
            try:
                df = pd.read_csv(newfolder+'/'+str(seed)+"_vals.csv")
                dfs_cat.append(df)
            except:
                print(4,eta,obj,seed)
        running_ind += 1
dfs_cat = pd.concat(dfs_cat)

dfs_lcx = []
running_ind = 0
newfolder = foldername6+str(running_ind)
for seed in seeds1:
    for comp in range(20):
        try:
            df = pd.read_csv(newfolder+'/'+str(seed)+'_'+str(comp)+"_vals_lcx.csv")
            dfs_lcx.append(df)
        except:
            print(6,comp,seed)
dfs_lcx = pd.concat(dfs_lcx)

collist = list(dfs_lcx.columns)
grouped = dfs_lcx.groupby(["seed"], as_index=False)
mean_vals = grouped[collist].mean()
df_lcx = mean_vals
inds = {}
#target_list = [0.01,0.05,0.1,0.15,0.20]
target_list = [0,0.01,0.02,0.03,0.05,0.08,0.1,0.12,0.15,0.18,0.20]
dfs_best = {}
dif=0.01
curdif = dif
for target in target_list:
    inds[target] = []
    dfs_best[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target))
            if mindif >= dif:
                curdif = mindif
                print(curdif)
            best_idx = np.argmin(dfs_cat[dfs_cat["seed"] == seed][abs(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)<=curdif]["valid_obj"])
            inds[target].append(best_idx)
            cur_df = dfs_cat[dfs_cat["seed"] == seed][abs(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best[target].append(cur_df)
            # best_idx = np.argmin(np.abs(np.array(dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target)))
            # inds[target].append(best_idx)
            # cur_df = dfs_cat[dfs_cat["seed"] == seed].iloc[best_idx:best_idx+1]
            # dfs_best[target].append(cur_df)
        except:
            print(seed)
    dfs_best[target] = pd.concat(dfs_best[target])   
dfs_best_o = {}
dif=0.001
curdif = dif
for target in target_list:
    inds[target] = []
    dfs_best_o[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(dfs_cat_o[dfs_cat_o["seed"] == seed]["valid_prob"] - target))
            if mindif >= dif:
                curdif = mindif
                print(curdif)
            best_idx = np.argmin(dfs_cat_o[dfs_cat_o["seed"] == seed][abs(dfs_cat_o[dfs_cat_o["seed"] == seed]["valid_prob"] - target)<=curdif]["valid_obj"])
            inds[target].append(best_idx)
            cur_df = dfs_cat_o[dfs_cat_o["seed"] == seed][abs(dfs_cat_o[dfs_cat_o["seed"] == seed]["valid_prob"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_o[target].append(cur_df)
        except:
            print(seed)
    dfs_best_o[target] = pd.concat(dfs_best_o[target])   
dfs_best_mv = {}
for target in target_list:
    dfs_best_mv[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_mv[df_mv["seed"] == seed][abs(df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target)<=curdif]["Validate_val"])
            cur_df = df_mv[df_mv["seed"] == seed][abs(df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_mv[target].append(cur_df)
        except:
            print(seed)
    dfs_best_mv[target] = pd.concat(dfs_best_mv[target])   
dfs_best_pre = {}
for target in target_list:
    dfs_best_pre[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_pre[df_pre["seed"] == seed][abs(df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target)<=curdif]["Validate_val"])
            cur_df = df_pre[df_pre["seed"] == seed][abs(df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_pre[target].append(cur_df)
        except:
            print(seed)
    dfs_best_pre[target] = pd.concat(dfs_best_pre[target])  

dfs_best_dro = {}
for target in target_list:
    dfs_best_dro[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_dro[df_dro["seed"] == seed]["Avg_prob_validate"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_dro[df_dro["seed"] == seed][abs(df_dro[df_dro["seed"] == seed]["Avg_prob_validate"] - target)<=curdif]["Validate_val"])
            cur_df = df_dro[df_dro["seed"] == seed][abs(df_dro[df_dro["seed"] == seed]["Avg_prob_validate"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_dro[target].append(cur_df)
        except:
            print(seed)
    dfs_best_dro[target] = pd.concat(dfs_best_dro[target])  

dfs_best_lcx = {}
for target in target_list:
    dfs_best_lcx [target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_lcx[df_lcx["seed"] == seed][abs(df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target)<=curdif]["valid_lcx_obj"])
            cur_df = df_lcx[df_lcx["seed"] == seed][abs(df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target)<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_lcx[target].append(cur_df)
        except:
            print(seed)
    dfs_best_lcx[target] = pd.concat(dfs_best_lcx[target])  
plot_data = []
for target in target_list:
    data = {'target':target,'test_prob': dfs_best[target]["test_prob"].mean(),'test_obj': dfs_best[target]["test_obj"].mean(),'0.25_test_obj': dfs_best[target]["test_obj"].quantile(0.25), '0.75_test_obj': dfs_best[target]["test_obj"].quantile(0.75),'test_prob_o': dfs_best_o[target]["test_prob"].mean(),'test_obj_o': dfs_best_o[target]["test_obj"].mean(),'0.25_test_obj_o': dfs_best_o[target]["test_obj"].quantile(0.25), '0.75_test_obj_o': dfs_best_o[target]["test_obj"].quantile(0.75),'mv_prob': dfs_best_mv[target]["Avg_prob_test"].mean(),'mv_obj':dfs_best_mv[target]["Test_val"].mean(), '0.25_mv_obj':dfs_best_mv[target]["Test_val"].quantile(0.25),  '0.75_mv_obj':dfs_best_mv[target]["Test_val"].quantile(0.75),'pre_prob': dfs_best_pre[target]["Avg_prob_test"].mean(),'pre_obj':dfs_best_pre[target]["Test_val"].mean(), '0.25_pre_obj':dfs_best_pre[target]["Test_val"].quantile(0.25),  '0.75_pre_obj':dfs_best_pre[target]["Test_val"].quantile(0.75),'dro_prob': dfs_best_dro[target]["Avg_prob_test"].mean(),'dro_obj':dfs_best_dro[target]["Test_val"].mean(), '0.25_dro_obj':dfs_best_dro[target]["Test_val"].quantile(0.25),  '0.75_dro_obj':dfs_best_dro[target]["Test_val"].quantile(0.75),"nonrob_prob":df_nonrob["nonrob_prob"].mean(), "nonrob_obj": df_nonrob["nonrob_obj"].mean(), "scenario_prob":df_nonrob["scenario_probs"].mean(), "scenario_obj": df_nonrob["scenario_obj"].mean(),'0.25_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.25),  '0.75_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.75), '0.25_scenario_obj':df_nonrob["scenario_obj"].quantile(0.25),  '0.75_scenario_obj':df_nonrob["scenario_obj"].quantile(0.75),'lcx_prob': dfs_best_lcx[target]["test_lcx_prob"].mean(),'lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].mean(), '0.25_lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].quantile(0.25),  '0.75_lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].quantile(0.75),
            "test_avg": dfs_best[target]["avg_val"].mean(), "test_avg_o": dfs_best_o[target]["avg_val"].mean(), "test_avg_mv": dfs_best_mv[target]["Test_other_obj"].mean(), "test_avg_pre": dfs_best_pre[target]["Test_other_obj"].mean(), "test_avg_lcx": dfs_best_lcx[target]["test_avg"].mean(), "test_avg_dro": dfs_best_dro[target]["Test_other_obj"].mean(),"test_avg_scene":df_nonrob["scenario_avg"].mean()
            }
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
    plt.plot(np.array(plot_data["mv_prob"]),np.array(plot_data["mv_obj"]), label = "MV test" ,color = "tab:blue",marker = "^")
    plt.fill_between(np.array(plot_data["mv_prob"]),np.array(plot_data["0.25_mv_obj"]),np.array(plot_data["0.75_mv_obj"]),alpha = 0.25, color = "tab:blue")
    
    plt.plot(np.array(plot_data["test_prob"]),np.array(plot_data["test_obj"]), label = "Trained test" ,color = "tab:orange",marker = "D")
    plt.fill_between(np.array(plot_data["test_prob"]),np.array(plot_data["0.25_test_obj"]),np.array(plot_data["0.75_test_obj"]),alpha = 0.25, color = "tab:orange")

    plt.plot(np.array(plot_data["test_prob_o"]),np.array(plot_data["test_obj_o"]), label = "Trained test delage" ,color = "tab:brown",marker = "D")
    plt.fill_between(np.array(plot_data["test_prob_o"]),np.array(plot_data["0.25_test_obj_o"]),np.array(plot_data["0.75_test_obj_o"]),alpha = 0.25, color = "tab:brown")

    plt.plot(np.array(plot_data["pre_prob"]),np.array(plot_data["pre_obj"]), label = "Pretrained test" ,color = "tab:green",marker = "v")
    plt.fill_between(np.array(plot_data["pre_prob"]),np.array(plot_data["0.25_pre_obj"]),np.array(plot_data["0.75_pre_obj"]),alpha = 0.25, color = "tab:green")
    
    plt.plot(np.array(plot_data["dro_prob"]),np.array(plot_data["dro_obj"]), label = "DRO test" ,color = "tab:olive",marker = "v")
    plt.fill_between(np.array(plot_data["dro_prob"]),np.array(plot_data["0.25_dro_obj"]),np.array(plot_data["0.75_dro_obj"]),alpha = 0.25, color = "tab:olive")

    plt.plot(np.array(plot_data["lcx_prob"]),np.array(plot_data["lcx_obj"]), label = "LCX test" ,color = "tab:pink",marker = "s")
    plt.fill_between(np.array(plot_data["lcx_prob"]),np.array(plot_data["0.25_lcx_obj"]),np.array(plot_data["0.75_lcx_obj"]),alpha = 0.25, color = "tab:pink")

    # plt.scatter(df_nonrob["nonrob_prob"].mean(),df_nonrob["nonrob_obj"].mean(),color = "tab:pink",marker = "s")
    # plt.fill_between(np.array(dfs[idx]["mean_nonrob_prob"])[1:],np.array(dfs[idx]["0.25_nonrob_obj"])[1:],np.array(dfs[idx]["0.75_nonrob_obj"])[1:],alpha = 0.25,color = "tab:pink")
    plt.plot(df_nonrob["scenario_probs"].mean(),df_nonrob["scenario_obj"].mean(),label = "Scenario",color = "tab:purple",marker = "o")
    # plt.fill_between(np.array(dfs[idx]["mean_scenario_probs"])[1:],np.array(dfs[idx]["0.25_scenario_obj"])[1:],np.array(dfs[idx]["0.75_scenario_obj"])[1:],alpha = 0.25,color = "tab:purple")
    plt.vlines(target_list,ymin = np.min(plot_data["test_obj"]),ymax=np.max(plot_data["test_obj"]),linestyles=":",color = "red",alpha = 0.5)

    # plt.vlines([0.03,0.05,0.10],ymin = -6,ymax=-2,linestyles=":",color = "red",alpha = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("Probability of constraint violation")
    # plt.ylabel("Out-of-sample objective")
    plt.ylim(ylim)
    plt.tight_layout()
    # plt.xlim([-0.02,0.20])
    plt.title("Out-of-sample objectives (test set)")
    plt.savefig(path+"Test_objectives.pdf",bbox_inches='tight')

def plot_best_avg(plot_data,ylim=None):
    idx = 1
    plt.figure(figsize = (8,4))
    plt.plot(np.array(plot_data["mv_prob"]),np.array(plot_data["test_avg_mv"]), label = "MV test" ,color = "tab:blue",marker = "^")
    
    plt.plot(np.array(plot_data["test_prob"]),np.array(plot_data["test_avg"]), label = "Trained test" ,color = "tab:orange",marker = "D")

    plt.plot(np.array(plot_data["test_prob_o"]),np.array(plot_data["test_avg_o"]), label = "Trained test delage" ,color = "tab:brown",marker = "D")

    plt.plot(np.array(plot_data["pre_prob"]),np.array(plot_data["test_avg_pre"]), label = "Pretrained test" ,color = "tab:green",marker = "v")
    
    plt.plot(np.array(plot_data["dro_prob"]),np.array(plot_data["test_avg_dro"]), label = "DRO test" ,color = "tab:olive",marker = "v")

    plt.plot(np.array(plot_data["lcx_prob"]),np.array(plot_data["test_avg_lcx"]), label = "LCX test" ,color = "tab:pink",marker = "s")

    plt.plot(np.array(plot_data["scenario_prob"]),np.array(plot_data["test_avg_scene"]),label = "Scenario test",color = "tab:purple",marker = "o")

    plt.legend(loc = "upper right")
    plt.xlabel("Probability of constraint violation")
    # plt.ylabel("Out-of-sample objective")
    plt.ylim(ylim)
    plt.tight_layout()
    # plt.xlim([-0.02,0.20])
    plt.title("Out-of-sample averages (test set)")
    plt.savefig(path+"Test_avg.pdf",bbox_inches='tight')
plot_best(plot_data,dfs,dfs_grid,dfs_mv_grid)
plot_best_avg(plot_data)
