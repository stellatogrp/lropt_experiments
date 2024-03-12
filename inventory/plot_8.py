import argparse
import os
import sys
output_stream = sys.stdout

import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":16,
    "font.family": "serif"
})

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                        default="portfolio/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

nvals = np.array([100])
m = 8
lower_q = 0.3
upper_q = 0.7
etas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
testetas = [0, 0.001, 0.002, 0.003, 0.004, 0.005,0.008, 0.01, 0.02, 0.0275, 0.03, 0.04,0.0465, 0.05, 0.06, 0.066, 0.09, 0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.25,0.30]
val_st = {}
val_re = {}
val_st_lower = {}
val_re_lower = {}
val_st_upper = {}
val_re_upper = {}
val_ro = {}
val_ro_lower = {}
prob_ro = {}
val_ro_upper = {}
prob_st = {}
prob_re = {}
val_re_nom_upper = {}
val_re_nom_lower = {}
val_re_nom = {}
prob_re_nom = {}
mu_vals = {}
lam_vals = {}
for N in nvals:
    mu_vals[N] = []
    lam_vals[N] = []
    val_st[N] = []
    val_re[N] = []
    prob_st[N] = []
    prob_re[N] = []
    val_st_lower[N] = []
    val_re_lower[N] = []
    val_ro[N] = []
    prob_ro[N] = []
    val_ro_lower[N] = []
    val_ro_upper[N] = []
    val_st_upper[N] = []
    val_re_upper[N] = []
    val_re_nom[N] = []
    val_re_nom_upper[N] = []
    val_re_nom_lower[N] = []
    prob_re_nom[N] = []
    offset = 8
    for i in range(len(etas)):
        print(etas[i])
        dftrain = pd.read_csv(foldername + f"results{i+offset}/" + f"train_{N,m,0}.csv")
        shape = dftrain.shape[0]
        lam_vals[N] = [val[0] for val in dftrain['lam_list']]
        mu_vals[N] = np.array(dftrain['mu'])
        values_st = []
        values_re = []
        values_re2 = []
        tp_prob_st = []
        tp_prob_re = []
        tp_prob_st0 = []
        tp_prob_re0 = []
        tp_prob_re2 = []
        values_ro = []
        tp_prob_ro = []
        probs_beta = {}
        for method in range(6):
            probs_beta[method] = []
        for r in range(20):
            dfgrid3 = pd.read_csv(foldername + f"results{0}/" + f"gridmv_{N,m,r}.csv")
            dfgrid2 = pd.read_csv(foldername + f"results{i}/" + f"gridre_{N,m,r}.csv")
            dfgrid = pd.read_csv(foldername + f"results{8}/" + f"gridmv_{N,m,r}.csv")
            ind_s = [np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            values_st.append(np.array(dfgrid['Test_val'][ind_s]))
            tp_prob_st.append(np.array(dfgrid['Avg_prob_test'][ind_s]))
            probs_beta[0].append(tp_prob_st[-1]>= 0.03)
            probs_beta[3].append(tp_prob_st[-1]>= 0.05)

            ind_2 = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-1).argmin()
            ind_r = [np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_ro = [np.absolute(np.mean(np.vstack(dfgrid3['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            
            values_re.append(np.array(dfgrid2['Test_val'][ind_r]))
            tp_prob_re.append(np.array(dfgrid2['Avg_prob_test'][ind_r]))
            values_re2.append(dfgrid2['Test_val'][ind_2])
            tp_prob_re2.append(dfgrid2['Avg_prob_test'][ind_2])
            values_ro.append(np.array(dfgrid3['Test_val'][ind_ro]))
            tp_prob_ro.append(np.array(dfgrid3['Avg_prob_test'][ind_ro]))  

            probs_beta[1].append(tp_prob_re[-1]>= 0.03)
            probs_beta[2].append(tp_prob_ro[-1]>= 0.03)
            probs_beta[4].append(tp_prob_re[-1]>= 0.05)
            probs_beta[5].append(tp_prob_ro[-1]>= 0.05)

        val_st_temp = np.vstack(values_st)
        val_re_temp = np.vstack(values_re)
        prob_st_temp = np.vstack(tp_prob_st)
        prob_re_temp = np.vstack(tp_prob_re)
        val_ro_temp = np.vstack(values_ro)
        prob_ro_temp = np.vstack(tp_prob_ro)
        probs_beta[0] = np.vstack(probs_beta[0])
        probs_beta[1] = np.vstack(probs_beta[1])
        probs_beta[2] = np.vstack(probs_beta[2])
        
        print("st", np.mean(probs_beta[0],axis=0))
        print("re", np.mean(probs_beta[1],axis=0))
        print("ro", np.mean(probs_beta[2],axis=0))
        print("st1", np.mean(probs_beta[3],axis=0))
        print("re1", np.mean(probs_beta[4],axis=0))
        print("ro1", np.mean(probs_beta[5],axis=0))

        val_ro[N].append(np.mean(val_ro_temp,axis=0))
        prob_ro[N].append(np.mean(prob_ro_temp,axis=0))
        val_re[N].append(np.mean(val_re_temp,axis=0))
        val_st[N].append(np.mean(val_st_temp,axis=0))
        prob_re[N].append(np.mean(tp_prob_re,axis=0))
        prob_st[N].append(np.mean(tp_prob_st,axis=0))
        val_st_lower[N].append(np.quantile(val_st_temp,lower_q,axis=0))
        val_st_upper[N].append(np.quantile(val_st_temp,upper_q,axis=0))
        val_re_lower[N].append(np.quantile(val_re_temp,lower_q,axis=0))
        val_re_upper[N].append(np.quantile(val_re_temp,upper_q,axis=0))
        val_ro_upper[N].append(np.quantile(val_ro_temp,upper_q,axis=0))
        val_ro_lower[N].append(np.quantile(val_ro_temp,lower_q,axis=0))

        val_re_nom_upper[N].append(np.quantile(values_re2,upper_q))
        val_re_nom_lower[N].append(np.quantile(values_re2,lower_q))
        val_re_nom[N].append(np.mean(values_re2))
        prob_re_nom[N].append(np.mean(tp_prob_re2))

    val_re[N] = np.vstack(val_re[N])
    val_re_lower[N] = np.vstack(val_re_lower[N])
    val_re_upper[N] = np.vstack(val_re_upper[N])
    val_st_lower[N] = np.vstack(val_st_lower[N])
    val_st_upper[N] = np.vstack(val_st_upper[N])

    val_ro[N] = np.vstack(val_ro[N])
    val_ro_lower[N] = np.vstack(val_ro_lower[N])
    val_ro_upper[N] = np.vstack(val_ro_upper[N])
    prob_ro[N] = np.vstack(prob_ro[N])
    prob_re[N] = np.vstack(prob_re[N])
    val_st[N] = np.vstack(val_st[N])
    prob_st[N] = np.vstack(prob_st[N])
    inds_re = np.argmin(val_re[N],axis = 0)
    inds_st = np.argmin(val_st[N],axis = 0)
    inds_ro = np.argmin(val_ro[N],axis = 0)
    
    print(inds_re, inds_st, inds_ro)

    val_re_plot = []
    prob_re_plot = []
    val_re_lower_plot = []
    val_re_upper_plot = []
    for ind_val in range(len(testetas)):
        candidate_prob = []
        candidate_val = []
        candidate_lower = []
        candidate_upper = []
        for ind_val_2 in range(len(etas)):
            if prob_re[N][ind_val_2][ind_val] <= testetas[ind_val]+0.001:
                candidate_prob.append(prob_re[N][ind_val_2][ind_val])
                candidate_val.append(val_re[N][ind_val_2][ind_val])
                candidate_lower.append(val_re_lower[N][ind_val_2][ind_val])
                candidate_upper.append(val_re_upper[N][ind_val_2][ind_val])
        if len(candidate_val) >= 1:
            min_ind = np.argmin(candidate_val)
        else:
            min_ind = 0
            candidate_val = [val_re[N][0][ind_val]]
            candidate_prob = [prob_re[N][0][ind_val]]
            candidate_lower = [val_re_lower[N][0][ind_val]]
            candidate_upper = [val_re_upper[N][0][ind_val]]
        print(ind_val, testetas[ind_val], min_ind)
        val_re_plot.append(candidate_val[min_ind])
        prob_re_plot.append(candidate_prob[min_ind])
        val_re_lower_plot.append(candidate_lower[min_ind])
        val_re_upper_plot.append(candidate_upper[min_ind])

    # val_re_plot = [val_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    # prob_re_plot = [prob_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_plot = [val_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    prob_st_plot = [prob_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_ro_plot = [val_ro[N].T[i][inds_re[i]] for i in range(len(testetas))]
    prob_ro_plot = [prob_ro[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_lower_plot = [val_st_lower[N].T[i][inds_st[i]] for i in range(len(testetas))]
    # val_re_lower_plot = [val_re_lower[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_upper_plot = [val_st_upper[N].T[i][inds_st[i]] for i in range(len(testetas))]
    # val_re_upper_plot = [val_re_upper[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_ro_lower_plot = [val_ro_lower[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_ro_upper_plot = [val_ro_upper[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    print("ro ", prob_ro_plot, val_ro_plot)
    print("nom ", prob_re_nom[N],val_re_nom[N])
    print("dro ", prob_st_plot, val_st_plot)
    print("Re ", prob_re_plot, val_re_plot)
    
    plt.figure(figsize = (6,3))

    plt.plot(prob_ro_plot, val_ro_plot, label = "Mean-Var RO", color = "tab:blue" )
    plt.fill_between(prob_ro_plot,val_ro_lower_plot,val_ro_upper_plot, color = "tab:blue", alpha=0.3)

    plt.plot(prob_re_plot, val_re_plot, label = "Reshaped RO", color = "tab:orange")
    plt.fill_between(prob_re_plot,val_re_lower_plot,val_re_upper_plot, color = "tab:orange", alpha=0.3)

    plt.plot(prob_st_plot, val_st_plot, label = "Wass DRO", color = "tab:green")

    plt.fill_between(prob_st_plot,val_st_lower_plot,val_st_upper_plot, color = "tab:green", alpha=0.3)

    plt.ylim([-466,-456.5])
    plt.vlines(ymin=-466, ymax=-456.5, x=0.028, linestyles=":",
           color="tab:red", label=r"$\hat{\eta}=0.028,0.064$") 
    plt.vlines(ymin=-466, ymax=-456.5, x=0.064, linestyles=":",
           color="tab:red") 
    plt.hlines(xmin=0.028, xmax=0.064, y=-461.51, linestyles="--",
           color="black") 
    
    plt.xlabel(r"Prob. of constraint violation $(\hat{\eta})$")
    plt.ylabel("Objective value")
    plt.title(f"$m={m}$")
    plt.legend()
    plt.savefig(foldername + f"{m}_{N}.pdf", bbox_inches='tight')
    plt.show()

