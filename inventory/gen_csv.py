import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
path = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/lropt_experiments/inventory_parallel/plots2/"
R = 10
etas = [0.01,0.05,0.10,0.12,0.15,0.20]
objs = [0.5,1,2]
seeds1 = [0,10,20,30,40,50,60,70,80,90]
foldername1 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/inv_results/worst_new/cvar/1.5/"
foldername2 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/inv_results/worst_new/cvar/1.5/"
foldername3 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/inv_results/worst_new/dro_sep/30/"
foldername4 = "/Users/irina.wang/Desktop/Princeton/Project2/lropt_experiments/inv_results/worst/dro/30/"

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
        None
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
df_mv = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_mean_var_grid.csv")
        df['seed'] = seeds1[seed]
        df_mv.append(df)
    except:
        None
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
df_dro1 = []
running_ind = 0
newfolder = foldername4+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_dro_grid.csv")
        df['seed'] = seeds1[seed]
        df_dro1.append(df)
    except:
        print(3,seed)
df_dro1 = pd.concat(df_dro1)
df_dro = []
running_ind = 0
newfolder = foldername3+str(running_ind)
for seed in range(R):
    df_dro_temp = []
    for context in range(10):
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


df_nonrob = []
running_ind = 0
newfolder = foldername1+str(running_ind)
for seed in range(R):
    try:
        df = pd.read_csv(newfolder+'/'+str(seed)+"_vals_nonrob.csv")
        df_nonrob.append(df)
    except:
        None
df_nonrob = pd.concat(df_nonrob)

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

inds = {}
target_list = [0.083,0.09,0.1]
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
plot_data = []
for target in target_list:
    data = {'target':target,
            'test_prob': dfs_best[target]["test_prob"].mean(),
            'test_obj': dfs_best[target]["test_obj"].mean(),
            '0.25_test_obj': dfs_best[target]["test_obj"].quantile(0.25), 
            '0.75_test_obj': dfs_best[target]["test_obj"].quantile(0.75),'mv_prob': dfs_best_mv[target]["Avg_prob_test"].mean(),'mv_obj':dfs_best_mv[target]["Test_worst"].mean(), 
            '0.25_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.25),  
            '0.75_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.75),'pre_prob': dfs_best_pre[target]["Avg_prob_test"].mean(),'pre_obj':dfs_best_pre[target]["Test_worst"].mean(), 
            '0.25_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.25),  '0.75_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.75),'dro_prob': dfs_best_dro[target]["Avg_prob_test"].mean(),'dro_obj':dfs_best_dro[target]["Test_worst"].mean(), 
            '0.25_dro_obj':dfs_best_dro[target]["Test_worst"].quantile(0.25),  '0.75_dro_obj':dfs_best_dro[target]["Test_worst"].quantile(0.75),
            'dro_prob_aggr': dfs_best_dro1[target]["Avg_prob_test"].mean(),'dro_obj_aggr':dfs_best_dro1[target]["Test_worst"].mean(), 
            '0.25_dro_obj_aggr':dfs_best_dro1[target]["Test_worst"].quantile(0.25),  
            '0.75_dro_obj_aggr':dfs_best_dro1[target]["Test_worst"].quantile(0.75), 
            "nonrob_prob":df_nonrob["nonrob_prob"].mean(), 
            "nonrob_obj": df_nonrob["nonrob_obj"].mean(), "scenario_prob":df_nonrob["scenario_probs"].mean(), 
            "scenario_obj": df_nonrob["scenario_obj"].mean(),
            '0.25_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.25),  
            '0.75_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.75), 
            '0.25_scenario_obj':df_nonrob["scenario_obj"].quantile(0.25),  
            '0.75_scenario_obj':df_nonrob["scenario_obj"].quantile(0.75),"test_avg": dfs_best[target]["avg_val"].mean(), 
            "0.25_test_avg": dfs_best[target]["avg_val"].quantile(0.25),
            "0.75_test_avg": dfs_best[target]["avg_val"].quantile(0.75), "test_avg_mv": dfs_best_mv[target]["Test_val"].mean(), 
            "0.25_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.25),"0.75_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.75),"test_avg_pre": dfs_best_pre[target]["Test_val"].mean(), 
            "0.25_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.75),
            "test_avg_dro": dfs_best_dro[target]["Test_val"].mean(),
            "0.25_test_avg_dro": dfs_best_dro[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_dro": dfs_best_dro[target]["Test_val"].quantile(0.75),
            "test_avg_dro_aggr": dfs_best_dro1[target]["Test_val"].mean(),
            "0.25_test_avg_dro_aggr": dfs_best_dro1[target]["Test_val"].quantile(0.25),
            "0.75_test_avg_dro_aggr": dfs_best_dro1[target]["Test_val"].quantile(0.75),
            "test_avg_scene":df_nonrob["scenario_avg"].mean(),
            "0.25_test_avg_scene":df_nonrob["scenario_avg"].quantile(0.25),
            "0.75_test_avg_scene":df_nonrob["scenario_avg"].quantile(0.75),
            "test_cvar": dfs_best[target]["test_cvar"].mean(),
            'test_mv_cvar':dfs_best_mv[target]["Test_cvar"].mean(), 
            '0.25_mv_cvar':dfs_best_mv[target]["Test_cvar"].quantile(0.25),  
            '0.75_mv_cvar':dfs_best_mv[target]["Test_cvar"].quantile(0.75),
            'test_pre_cvar':dfs_best_pre[target]["Test_cvar"].mean(),  
            '0.25_pre_cvar':dfs_best_pre[target]["Test_cvar"].quantile(0.25),  '0.75_pre_cvar':dfs_best_pre[target]["Test_cvar"].quantile(0.75),'test_dro_cvar':dfs_best_dro[target]["Test_cvar"].mean(), 
            '0.25_dro_cvar':dfs_best_dro[target]["Test_cvar"].quantile(0.25),  '0.75_dro_cvar':dfs_best_dro[target]["Test_cvar"].quantile(0.75),
            "test_scene_cvar":df_nonrob["scenario_cvar"].mean(),'test_dro_cvar_aggr':dfs_best_dro1[target]["Test_cvar"].mean(), 
            '0.25_dro_cvar_aggr':dfs_best_dro1[target]["Test_cvar"].quantile(0.25),  
            '0.75_dro_cvar_aggr':dfs_best_dro1[target]["Test_cvar"].quantile(0.75),
            }
    data = pd.DataFrame(data, index=[0])
    plot_data.append(data)
plot_data = pd.concat(plot_data)
plot_data.to_csv(path+"plot_data.csv") 
