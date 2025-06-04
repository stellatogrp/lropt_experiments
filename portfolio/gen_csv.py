import numpy as np
import pandas as pd
import sys
import os
sys.path.append('..')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int,
                    default=2000)
parser.add_argument('--n', type=int,
                    default=10)
arguments = parser.parse_args()
N = arguments.N
n = arguments.n

path = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/port_parallel/plots/plots_4/{N}_{n}/"

if not os.path.exists(path):
    os.makedirs(path)
R = 10
etas = [0.05,0.08,0.10,0.12,0.15,0.2]
objs = [1,1.5,2.5,3,5,8]
etas1 = [0.05,0.1,0.12,0.15,0.2,0.3]
objs1 = [0.9,0.6,0.5,0.4,0.2,0.1]
seeds1 = [0,10,20,30,40,50,60,70,80,90]
foldername1 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/fixed/{N}_{n}/"
foldername2 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/bi/{N}_{n}/"
foldername3 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/dro_sep/{N}_{n}/"
foldername4 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/delage/{N}_{n}/"
foldername5 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/lcx/{N}_{n}/"
foldername6 = f"/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/port_results/worst/dro/{N}_{n}/"
quantiles = [0.25,0.75]

df_pre = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_linear_pretrained_grid.csv")
        df['seed'] = seeds1[seed]
        df_pre.append(df)
    except:
        print(2,seed)
df_pre = pd.concat(df_pre)

df_mv = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_mean_var_grid.csv")
        df['seed'] = seeds1[seed]
        df_mv.append(df)
    except:
        print(3,seed)
df_mv = pd.concat(df_mv)
dfs_mv = []

df_dro = []
running_ind = 0
newfolder = foldername3+str(running_ind)
for seed in range(R):
    df_dro_temp = []
    for context in range(20):
        try:
            df = pd.read_csv(newfolder+'/'+str(seed)+'_'+str(context)+"_dro_grid.csv")
            df['seed'] = seeds1[seed]
            df_dro_temp.append(df)
        except:
            print(3,seed)
    df_dro_temp = pd.concat(df_dro_temp)
    collist = list(df_dro_temp.columns)
    collist.remove("Probability_violations_validate")
    collist.remove('Probability_violations_test')
    collist.remove("step")
    grouped = df_dro_temp.groupby(["Rho"], as_index=False)
    df_dro.append(grouped[collist].mean())
df_dro = pd.concat(df_dro)

df_dro1 = []
running_ind = 0
newfolder = foldername6+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_dro_grid.csv")
        df['seed'] = seeds1[seed]
        df_dro1.append(df)
    except:
        print(3,seed)
df_dro1 = pd.concat(df_dro1)

df_nonrob = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_vals_nonrob.csv")
        df_nonrob.append(df)
    except:
        print(4,seed)
df_nonrob = pd.concat(df_nonrob)

dfs_cat_o = []
running_ind = 0
for eta in etas1:
    for obj in objs1:
        newfolder = foldername4+str(running_ind)
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
        newfolder = foldername2+str(running_ind)
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
newfolder = foldername5+str(running_ind)
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
target_list = [0.1]
dfs_best = {}
dif=0.0
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
            best_idx = np.argmin(dfs_cat[dfs_cat["seed"] == seed][dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target<=curdif]["valid_obj"])
            inds[target].append(best_idx)
            cur_df = dfs_cat[dfs_cat["seed"] == seed][dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best[target].append(cur_df)
        except:
            print(seed)
    dfs_best[target] = pd.concat(dfs_best[target])   
dfs_best_o = {}
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
            best_idx = np.argmin(dfs_cat_o[dfs_cat_o["seed"] == seed][dfs_cat_o[dfs_cat_o["seed"] == seed]["valid_prob"] - target<=curdif]["valid_obj"])
            inds[target].append(best_idx)
            cur_df = dfs_cat_o[dfs_cat_o["seed"] == seed][dfs_cat_o[dfs_cat_o["seed"] == seed]["valid_prob"] - target<=curdif].iloc[best_idx:best_idx+1]
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
            best_idx = np.argmin(df_mv[df_mv["seed"] == seed][df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_worst"])
            cur_df = df_mv[df_mv["seed"] == seed][df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
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
            best_idx = np.argmin(df_pre[df_pre["seed"] == seed][df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_worst"])
            cur_df = df_pre[df_pre["seed"] == seed][df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
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
            best_idx = np.argmin(df_dro[df_dro["seed"] == seed][df_dro[df_dro["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_worst"])
            cur_df = df_dro[df_dro["seed"] == seed][df_dro[df_dro["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_dro[target].append(cur_df)
        except:
            print(seed)
    dfs_best_dro[target] = pd.concat(dfs_best_dro[target])  

dfs_best_dro1 = {}
for target in target_list:
    dfs_best_dro1[target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_dro1[df_dro1["seed"] == seed]["Avg_prob_validate"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_dro1[df_dro1["seed"] == seed][df_dro1[df_dro1["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_worst"])
            cur_df = df_dro1[df_dro1["seed"] == seed][df_dro1[df_dro1["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_dro1[target].append(cur_df)
        except:
            print(seed)
    dfs_best_dro1[target] = pd.concat(dfs_best_dro1[target])  

dfs_best_lcx = {}
for target in target_list:
    dfs_best_lcx [target] = []
    for seed in seeds1:
        try:
            curdif = dif
            mindif = np.min(abs(df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target))
            if mindif >= dif:
                curdif = mindif
            best_idx = np.argmin(df_lcx[df_lcx["seed"] == seed][df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target<=curdif]["valid_lcx_obj"])
            cur_df = df_lcx[df_lcx["seed"] == seed][df_lcx[df_lcx["seed"] == seed]["valid_lcx_prob"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_lcx[target].append(cur_df)
        except:
            print(seed)
    dfs_best_lcx[target] = pd.concat(dfs_best_lcx[target])  
plot_data = []
for target in target_list:
    data = {'target':target,
            'test_prob': dfs_best[target]["test_prob"].mean(),
            'test_obj': dfs_best[target]["test_obj"].mean(),
            '0.25_test_obj': dfs_best[target]["test_obj"].quantile(0.25), 
            '0.75_test_obj': dfs_best[target]["test_obj"].quantile(0.75),
            'test_prob_o': dfs_best_o[target]["test_prob"].mean(),
            'test_obj_o': dfs_best_o[target]["test_obj"].mean(),
            '0.25_test_obj_o': dfs_best_o[target]["test_obj"].quantile(0.25), 
            '0.75_test_obj_o': dfs_best_o[target]["test_obj"].quantile(0.75),
            'mv_prob': dfs_best_mv[target]["Avg_prob_test"].mean(),
            'mv_obj':dfs_best_mv[target]["Test_worst"].mean(), 
            '0.25_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.25),  
            '0.75_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.75),
            'pre_prob': dfs_best_pre[target]["Avg_prob_test"].mean(),
            'pre_obj':dfs_best_pre[target]["Test_worst"].mean(), 
            '0.25_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.25),
            '0.75_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.75),
            'dro_prob': dfs_best_dro[target]["Avg_prob_test"].mean(),
            'dro_obj':dfs_best_dro[target]["Test_worst"].mean(), 
            '0.25_dro_obj':dfs_best_dro[target]["Test_worst"].quantile(0.25),
            '0.75_dro_obj':dfs_best_dro[target]["Test_worst"].quantile(0.75),
            'dro_prob_obj_aggr': dfs_best_dro1[target]["Avg_prob_test"].mean(),
            'dro_obj_obj_aggr':dfs_best_dro1[target]["Test_worst"].mean(), 
            '0.25_dro_obj_obj_aggr':dfs_best_dro1[target]["Test_worst"].quantile(0.25),  
            '0.75_dro_obj_obj_aggr':dfs_best_dro1[target]["Test_worst"].quantile(0.75),"scenario_prob":df_nonrob["scenario_probs"].mean(), 
            "scenario_obj": df_nonrob["scenario_obj"].mean(),
            '0.25_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.25),  
            '0.75_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.75), 
            '0.25_scenario_obj':df_nonrob["scenario_obj"].quantile(0.25),  
            '0.75_scenario_obj':df_nonrob["scenario_obj"].quantile(0.75),
            'lcx_prob': dfs_best_lcx[target]["test_lcx_prob"].mean(),
            'lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].mean(), 
            '0.25_lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].quantile(0.25),  
            '0.75_lcx_obj':dfs_best_lcx[target]["test_lcx_obj"].quantile(0.75),
            "test_avg_o": dfs_best_o[target]["avg_val"].mean(),
            "test_avg": dfs_best[target]["avg_val"].mean(), 
            "0.25_test_avg": dfs_best[target]["avg_val"].quantile(0.25),
            "0.75_test_avg": dfs_best[target]["avg_val"].quantile(0.75), "test_avg_mv": dfs_best_mv[target]["Test_val"].mean(), 
            "0.25_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.75),"test_avg_pre": dfs_best_pre[target]["Test_val"].mean(), 
            "0.25_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.75),
            "test_avg_lcx": dfs_best_lcx[target]["test_avg"].mean(),
            "test_avg_dro": dfs_best_dro[target]["Test_val"].mean(),
            "0.25_test_avg_dro": dfs_best_dro[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_dro": dfs_best_dro[target]["Test_val"].quantile(0.75),
            "test_avg_dro_obj_aggr": dfs_best_dro1[target]["Test_val"].mean(),
            "0.25_test_avg_dro_obj_aggr": dfs_best_dro1[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_dro_obj_aggr": dfs_best_dro1[target]["Test_val"].quantile(0.75),
            "test_avg_scene":df_nonrob["scenario_avg"].mean(),
            "0.25_test_avg_scene":df_nonrob["scenario_avg"].quantile(0.25),
            "0.75_test_avg_scene":df_nonrob["scenario_avg"].quantile(0.75),
            "test_cvar": dfs_best[target]["test_cvar"].mean(),
            "test_o_cvar": dfs_best_o[target]["test_cvar"].mean(),
            'test_mv_cvar':dfs_best_mv[target]["Test_cvar"].mean(), 
            '0.25_mv_cvar':dfs_best_mv[target]["Test_cvar"].quantile(0.25),  
            '0.75_mv_cvar':dfs_best_mv[target]["Test_cvar"].quantile(0.75),
            'test_pre_cvar':dfs_best_pre[target]["Test_cvar"].mean(),  
            '0.25_pre_cvar':dfs_best_pre[target]["Test_cvar"].quantile(0.25),  '0.75_pre_cvar':dfs_best_pre[target]["Test_cvar"].quantile(0.75),'test_dro_cvar':dfs_best_dro[target]["Test_cvar"].mean(), 
            '0.25_dro_cvar':dfs_best_dro[target]["Test_cvar"].quantile(0.25),  '0.75_dro_cvar':dfs_best_dro[target]["Test_cvar"].quantile(0.75),
            'test_dro_cvar_obj_aggr':dfs_best_dro1[target]["Test_cvar"].mean(), 
            '0.25_dro_cvar_obj_aggr':dfs_best_dro1[target]["Test_cvar"].quantile(0.25),  
            '0.75_dro_cvar_obj_aggr':dfs_best_dro1[target]["Test_cvar"].quantile(0.75),
            "test_scene_cvar":df_nonrob["scenario_cvar"].mean(),
            'test_lcx_cvar':dfs_best_lcx[target]["test_lcx_cvar"].mean(),  
            }
    data = pd.DataFrame(data, index=[0])
    plot_data.append(data)
plot_data = pd.concat(plot_data)
plot_data.to_csv(path+"plot_data.csv") 