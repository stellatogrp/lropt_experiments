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


def pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
    p_front.append(myList[-1])
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def pareto_frontier_3(Xs, Ys, Zs, maxX=False, maxY=False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_front.append(myList[-1])
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontZ = [pair[2] for pair in p_front]
    return p_frontX, p_frontY, p_frontZ


nvals = np.array([500])
n = 10
lower_q = 0.3
upper_q = 0.7
etas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
testetas = [0, 0.001, 0.002, 0.003, 0.004, 0.005,0.008, 0.01, 0.02, 0.0275, 0.03, 0.04, 0.0475, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.25,0.30]
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
val_rore = {}
val_rore_lower = {}
prob_rore = {}
val_rore_upper = {}
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
    val_rore[N] = []
    prob_rore[N] = []
    val_rore_lower[N] = []
    val_rore_upper[N] = []
    offset = 8
    for i in range(len(etas)):
        print(etas[i])
        dftrain = pd.read_csv(foldername + f"results{i+offset}/" + f"train_{N,n,0}.csv")
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
        values_rore = []
        tp_prob_rore = []
        probs_beta = {}
        for method in range(4):
            probs_beta[method] = []
        for r in range(20):
            dfgrid = pd.read_csv(foldername + f"results{16}/" + f"gridmv_{N,n,r}.csv")
            dfgrid2 = pd.read_csv(foldername  + f"results{i+offset}/" + f"gridre_{N,n,r}.csv")
            dfgrid3 = pd.read_csv(foldername + f"results{0}/" + f"gridmv_{N,n,r}.csv")
            dfgrid4 = pd.read_csv(foldername + f"results{i}/" + f"gridre_{N,n,r}.csv")

            ind_2 = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-1).argmin()
            ind_s = [np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_r = [np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_ro = [np.absolute(np.mean(np.vstack(dfgrid3['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_rore = [np.absolute(np.mean(np.vstack(dfgrid4['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            
            values_st.append(np.array(dfgrid['Test_val'][ind_s]))
            values_re.append(np.array(dfgrid2['Test_val'][ind_r]))
            tp_prob_st.append(np.array(dfgrid['Avg_prob_test'][ind_s]))
            tp_prob_re.append(np.array(dfgrid2['Avg_prob_test'][ind_r]))
            values_re2.append(dfgrid2['Test_val'][ind_2])
            tp_prob_re2.append(dfgrid2['Avg_prob_test'][ind_2])
            values_ro.append(np.array(dfgrid3['Test_val'][ind_ro]))
            tp_prob_ro.append(np.array(dfgrid3['Avg_prob_test'][ind_ro])) 
            values_rore.append(np.array(dfgrid4['Test_val'][ind_rore]))
            tp_prob_rore.append(np.array(dfgrid4['Avg_prob_test'][ind_rore]))  

            probs_beta[0].append(tp_prob_st[-1]>= 0.03)
            probs_beta[1].append(tp_prob_re[-1]>= 0.03)
            probs_beta[2].append(tp_prob_ro[-1]>= 0.03)
            probs_beta[3].append(tp_prob_rore[-1]>= 0.03)

        val_st_temp = np.vstack(values_st)
        val_re_temp = np.vstack(values_re)
        prob_st_temp = np.vstack(tp_prob_st)
        prob_re_temp = np.vstack(tp_prob_re)
        val_ro_temp = np.vstack(values_ro)
        prob_ro_temp = np.vstack(tp_prob_ro)
        val_rore_temp = np.vstack(values_rore)
        prob_rore_temp = np.vstack(tp_prob_rore)

        for method in range(4):
            probs_beta[method] = np.vstack(probs_beta[method])
            
        print("wass dro", np.mean(probs_beta[0],axis=0))
        print("reshaped mro", np.mean(probs_beta[1],axis=0))
        print("mv ro", np.mean(probs_beta[2],axis=0))
        print("reshaped ro", np.mean(probs_beta[3],axis=0))

        val_ro[N].append(np.mean(val_ro_temp,axis=0))
        prob_ro[N].append(np.mean(prob_ro_temp,axis=0))
        val_rore[N].append(np.mean(val_rore_temp,axis=0))
        prob_rore[N].append(np.mean(prob_rore_temp,axis=0))
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
        val_rore_upper[N].append(np.quantile(val_rore_temp,upper_q,axis=0))
        val_rore_lower[N].append(np.quantile(val_rore_temp,lower_q,axis=0))

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
    val_rore[N] = np.vstack(val_rore[N])
    val_rore_lower[N] = np.vstack(val_rore_lower[N])
    val_rore_upper[N] = np.vstack(val_rore_upper[N])
    prob_rore[N] = np.vstack(prob_rore[N])

    inds_re = np.argmin(val_re[N],axis = 0)
    inds_st = np.argmin(val_st[N],axis = 0)
    inds_ro = np.argmin(val_ro[N],axis = 0)
    inds_rore = np.argmin(val_rore[N],axis = 0)
    print(inds_re, inds_st, inds_ro, inds_rore)
    
    val_re_plot = [val_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    prob_re_plot = [prob_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_plot = [val_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    prob_st_plot = [prob_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_ro_plot = [val_ro[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    prob_ro_plot = [prob_ro[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_rore_plot = [val_rore[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    prob_rore_plot = [prob_rore[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    val_st_lower_plot = [val_st_lower[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_re_lower_plot = [val_re_lower[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_upper_plot = [val_st_upper[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_re_upper_plot = [val_re_upper[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_ro_lower_plot = [val_ro_lower[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_ro_upper_plot = [val_ro_upper[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_rore_lower_plot = [val_rore_lower[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    val_rore_upper_plot = [val_rore_upper[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    print("ro ", prob_ro_plot, val_ro_plot)
    print("nom ", prob_re_nom[N],val_re_nom[N])
    print("dro ", prob_st_plot, val_st_plot)
    print("Re ", prob_re_plot, val_re_plot)
    print("rore ", prob_rore_plot, val_rore_plot)


    # print(prob_re_nom[N],val_re_nom[N])
    # print(prob_st[N])

    # dfgrid = pd.read_csv(foldername + f"results{17}/" + f"gridmv_{N,n,0}.csv")
    # ro_probs = dfgrid["Avg_prob_test"]
    # ro_vals = dfgrid["Test_val"]
    # for r in range(1,20):
    #     dfgrid = pd.read_csv(foldername + f"results{17}/" + f"gridmv_{N,n,r}.csv")
    #     ro_probs = np.vstack([ro_probs, dfgrid["Avg_prob_test"]])
    #     ro_vals = np.vstack([ro_vals, dfgrid["Test_val"]])
    
    plt.figure(figsize = (6,3))

    plt.plot(prob_ro_plot, val_ro_plot, label = "Mean-Var RO", color = "tab:blue" )
    plt.fill_between(prob_ro_plot,val_ro_lower_plot,val_ro_upper_plot, color = "tab:blue", alpha=0.3)

    paretox, paretoy = pareto_frontier(prob_rore_plot[:],val_rore_plot[:])
    plt.plot(paretox, paretoy,label="Reshaped RO", color = "tab:orange")
    print("paretos", paretox, paretoy)
    paretox1, paretoylower, paretoyupper = pareto_frontier_3(prob_rore_plot[:],val_rore_lower_plot[:], val_rore_upper_plot[:])
    plt.fill_between(paretox1,paretoylower,paretoyupper, color = "tab:orange", alpha=0.3)

    plt.plot(prob_re_plot, val_re_plot, label = "Reshaped DRO", color = "tab:red")
    plt.fill_between(prob_re_plot,val_re_lower_plot,val_re_upper_plot, color = "tab:red", alpha=0.3)

    plt.plot(prob_st_plot, val_st_plot, label = "Wass DRO", color = "tab:green")

    plt.fill_between(prob_st_plot,val_st_lower_plot,val_st_upper_plot, color = "tab:green", alpha=0.3)
 
    plt.ylim([-0.86,-0.63])
    plt.vlines(ymin=-0.86, ymax=-0.63, x=0.03, linestyles=":",
           color="tab:red", label=r"$\hat{\eta}=0.03$") 
    plt.xlabel(r"Prob. of constraint violation $(\hat{\eta})$")
    plt.ylabel("Objective value")
    plt.title(f"$n={n}$")
    plt.legend(loc='upper right')
    plt.savefig(foldername + f"{N}_1.pdf", bbox_inches='tight')
    plt.show()

plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":18,
    "font.family": "serif"
})


dfgrid = pd.read_csv(foldername + f"results{16}/" + f"gridmv_{500,n,0}.csv")

dfgrid2 = pd.read_csv(foldername + f"results{15}/" + f"gridre_{500,n,0}.csv")
dfgrid3 = pd.read_csv(foldername + f"results{7}/" + f"gridmv_{500,n,0}.csv")
dfgrid4 = pd.read_csv(foldername  + f"results{7}/" + f"gridre_{N,n,12}.csv")

dros = []
ros = []
res = []
rores = []
print(np.array(dfgrid["Avg_prob_test"]))
print(np.array(dfgrid3["Avg_prob_test"]))
print(np.array(dfgrid2["Avg_prob_test"]))

for i in range(100):
    dro= np.array(dfgrid["var_values"])[i][11:-22]
    dro = dro[dro.index("[")+1:]
    dro = dro[:dro.index("]")]
    dro = dro.replace(" ","")
    dro = dro.replace("\n", "")
    dro = dro.split(",")
    ro = np.array(dfgrid3["var_values"])[i][11:-22]
    ro = ro[ro.index("[")+1:]
    ro = ro[:ro.index("]")]    
    ro = ro.replace(" ","")
    ro = ro.replace("\n", "")
    ro = ro.split(",")
    re = np.array(dfgrid2["var_values"])[i][11:-22]
    re = re[re.index("[")+1:]
    re = re[:re.index("]")]
    re = re.replace("\n", "")
    re = re.replace(" ","")
    re = re.split(",")
    rore = np.array(dfgrid4["var_values"])[i][11:-22]
    rore = rore[rore.index("[")+1:]
    rore = rore[:rore.index("]")]    
    rore = rore.replace(" ","")
    rore = rore.replace("\n", "")
    rore = rore.split(",")
    # print(i, dro,ro,re)
    ros.append(np.array([float(j) for j in ro]))
    res.append(np.array([float(j) for j in re]))
    dros.append(np.array([float(j) for j in dro]))
    rores.append(np.array([float(j) for j in rore]))


plt.figure(figsize = (5,4))
dros = np.vstack(dros)
for i in range(1, 11):
    plt.plot(np.array(dfgrid["Avg_prob_test"])[10:], np.sum(dros[10:, :i], axis=1),
               color='black', linewidth=1.0)
    plt.fill_between(np.array(dfgrid["Avg_prob_test"])[10:], np.sum(dros[10:, :i-1], axis=1), 
                       np.sum(dros[10:, :i], axis=1),color=plt.cm.RdYlBu(1 - i/11))
# plt.xlim([-0.03,0.33])
# plt.xscale("log")
plt.title("Wass DRO")
plt.xlabel(r"$\hat{\eta}$")
plt.ylabel("Portfolio weights")
plt.savefig(foldername + "Wass-dis.pdf", bbox_inches='tight')
plt.show()

plt.figure(figsize = (5,4))
ros = np.vstack(ros)
for i in range(1, 11):
    plt.plot(np.array(dfgrid3["Avg_prob_test"])[12:], np.sum(ros[12:, :i], axis=1),
               color='black', linewidth=1.0)
    plt.fill_between(np.array(dfgrid3["Avg_prob_test"])[12:], np.sum(ros[12:, :i-1], axis=1), 
                       np.sum(ros[12:, :i], axis=1),color=plt.cm.RdYlBu(1 - i/11))
# plt.xlim([-0.03,0.33])
# plt.xscale("log")
plt.title("Mean-Var RO")
plt.xlabel(r"$\hat{\eta}$")
plt.ylabel("Portfolio weights")
plt.savefig(foldername + "Mean-var-dis.pdf", bbox_inches='tight')
plt.show()

plt.figure(figsize = (5,4))
res = np.vstack(res)
for i in range(1, 11):
    plt.plot(np.array(dfgrid2["Avg_prob_test"]), np.sum(res[:, :i], axis=1),
               color='black', linewidth=1.0)
    plt.fill_between(np.array(dfgrid2["Avg_prob_test"]), np.sum(res[:, :i-1], axis=1), 
                       np.sum(res[:, :i], axis=1),color=plt.cm.RdYlBu(1 - i/11))
# plt.xlim([-0.03,0.33])
# plt.xscale("log")
plt.title("Reshaped DRO")
plt.xlabel(r"$\hat{\eta}$")
plt.ylabel("Portfolio weights")
plt.savefig(foldername + "Reshaped-dis.pdf", bbox_inches='tight')
plt.show()


plt.figure(figsize = (5,4))
rores = np.vstack(rores)
print(dfgrid4["Avg_prob_test"])
for i in range(1, 11):
    plt.plot(np.array(dfgrid4["Avg_prob_test"])[3:], np.sum(rores[3:, :i], axis=1),
               color='black', linewidth=1.0)
    plt.fill_between(np.array(dfgrid4["Avg_prob_test"])[3:], np.sum(rores[3:, :i-1], axis=1), 
                       np.sum(rores[3:, :i], axis=1),color=plt.cm.RdYlBu(1 - i/11))
# plt.xlim([-0.03,0.33])
# plt.xscale("log")
plt.title("Reshaped RO")
plt.xlabel(r"$\hat{\eta}$")
plt.ylabel("Portfolio weights")
plt.savefig(foldername + "Reshaped-ro-dis.pdf", bbox_inches='tight')
plt.show()