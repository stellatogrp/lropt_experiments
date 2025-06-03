import numpy as np
import pandas as pd
import sys
sys.path.append('..')
path = "news_intro/"
R = 10
etas = [0.05,0.10,0.12,0.15,0.20,0.25]
objs = [1,3,5,7,10]
seeds1 = [0,1,2,3,4,5,6,7,8,9]
foldername1 = "results/news/"
foldername2 = "results/news/"
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
target_list = [0.09,0.095,0.1]
dfs_best = {}
dif=0.002
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
            best_idx = np.argmin(dfs_cat[dfs_cat["seed"] == seed][dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target<=curdif]["avg_val"])
            inds[target].append(best_idx)
            cur_df = dfs_cat[dfs_cat["seed"] == seed][dfs_cat[dfs_cat["seed"] == seed]["valid_prob"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best[target].append(cur_df)
        except:
            print("trained",seed)
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
            best_idx = np.argmin(df_mv[df_mv["seed"] == seed][df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_val"])
            cur_df = df_mv[df_mv["seed"] == seed][df_mv[df_mv["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_mv[target].append(cur_df)
        except:
            print("mv",seed)
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
            best_idx = np.argmin(df_pre[df_pre["seed"] == seed][df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target<=curdif]["Validate_val"])
            cur_df = df_pre[df_pre["seed"] == seed][df_pre[df_pre["seed"] == seed]["Avg_prob_validate"] - target<=curdif].iloc[best_idx:best_idx+1]
            dfs_best_pre[target].append(cur_df)
        except:
            print("pre",seed)
    dfs_best_pre[target] = pd.concat(dfs_best_pre[target])  

plot_data = []
for target in target_list:
    data = {'target':target,'test_prob': dfs_best[target]["test_prob"].mean(),'test_obj': dfs_best[target]["test_obj"].mean(),'0.25_test_obj': dfs_best[target]["test_obj"].quantile(0.25), '0.75_test_obj': dfs_best[target]["test_obj"].quantile(0.75),'mv_prob': dfs_best_mv[target]["Avg_prob_test"].mean(),'mv_obj':dfs_best_mv[target]["Test_worst"].mean(), '0.25_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.25),  '0.75_mv_obj':dfs_best_mv[target]["Test_worst"].quantile(0.75),'pre_prob': dfs_best_pre[target]["Avg_prob_test"].mean(),'pre_obj':dfs_best_pre[target]["Test_worst"].mean(), '0.25_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.25),  '0.75_pre_obj':dfs_best_pre[target]["Test_worst"].quantile(0.75),"nonrob_prob":df_nonrob["nonrob_prob"].mean(), "nonrob_obj": df_nonrob["nonrob_obj"].mean(), "scenario_prob":df_nonrob["scenario_probs"].mean(), "scenario_obj": df_nonrob["scenario_obj"].mean(),'0.25_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.25),  '0.75_nonrob_obj':df_nonrob["nonrob_obj"].quantile(0.75), '0.25_scenario_obj':df_nonrob["scenario_obj"].quantile(0.25),  '0.75_scenario_obj':df_nonrob["scenario_obj"].quantile(0.75),
    "test_avg": dfs_best[target]["avg_val"].mean(), "0.25_test_avg": dfs_best[target]["avg_val"].quantile(0.25),"0.75_test_avg": dfs_best[target]["avg_val"].quantile(0.75), "test_avg_mv": dfs_best_mv[target]["Test_val"].mean(), "0.25_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.25),"0.75_test_avg_mv": dfs_best_mv[target]["Test_val"].quantile(0.75),
    "test_avg_pre": dfs_best_pre[target]["Test_val"].mean(), 
    "0.25_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.25),"0.75_test_avg_pre": dfs_best_pre[target]["Test_val"].quantile(0.75),
            "test_avg_scene":df_nonrob["scenario_avg"].mean()}
    data = pd.DataFrame(data, index=[0])
    plot_data.append(data)
plot_data = pd.concat(plot_data)
plot_data.to_csv(path+"plot_data.csv") 