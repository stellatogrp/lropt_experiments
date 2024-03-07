from utils import plot_tradeoff, plot_iters, plot_contours_line, plot_coverage_all
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import lropt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "text.usetex": True,

    "font.size": 24,
    "font.family": "serif"
})

import pandas as pd
data = pd.read_csv('https://xiongpengnus.github.io/rsome/taxi_rain.csv')
demand = data.loc[:, 'Region1':'Region10']  
data = demand.values                                    # sample demand as an array
I, J = 1, 10
r = np.array([4.50, 4.41, 3.61, 4.49, 4.38, 4.58, 4.53, 4.64, 4.58, 4.32])
c = 3 * np.ones(J)
q = 400 * np.ones(I)

scenarios = {}
num_scenarios = 1
for scene in range(num_scenarios):
    np.random.seed(scene)
    scenarios[scene] = {}
    scenarios[scene][0] = r


def plot_coverage_all(df_standard,df_reshape,dfs,title,ind_1 = (0,100), ind_2 = (0,100), logscale = True):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    beg1,end1 = ind_1
    beg2,end2 = ind_2

    fig, (ax, ax1,ax2) = plt.subplots(1, 3, figsize=(23, 3))
    
    ax.plot(np.mean(np.vstack(df_standard['Violations']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax.fill(np.append(np.quantile(np.vstack(df_standard['Violations']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Violations']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.9,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax.plot(np.mean(np.vstack(df_reshape['Violations']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color="tab:orange", label=r"Reshaped set")
    ax.fill(np.append(np.quantile(np.vstack(df_reshape['Violations']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Violations']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.9,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax.set_xlabel("Probability of constraint violation")
    ax.axvline(x = 0.03, color = "green", linestyle = "-.",label = r"$\eta = 0.03$")
    ax.set_ylabel("Objective value")
    ax.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    # ax.legend()

    ax1.plot(np.mean(np.vstack(df_standard['coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax1.fill(np.append(np.quantile(np.vstack(df_standard['coverage_test']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['coverage_test']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.90,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax1.plot(np.mean(np.vstack(df_reshape['coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",label=r"Decision-Focused set")
    ax1.fill(np.append(np.quantile(np.vstack(df_reshape['coverage_test']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['coverage_test']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.90,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    if dfs:
        for i in range(5):
            ax1.plot(np.mean(np.vstack(dfs[i+1][0]['coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(dfs[i+1][0]['Test_val']),axis = 1)[beg1:end1], color="tab:blue", linestyle = "-")
            ax1.plot(np.mean(np.vstack(dfs[i+1][1]['coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(dfs[i+1][1]['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",linestyle = "-")

    ax1.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax1.axvline(x = 0.8, color = "black", linestyle = ":",label = "0.8 Coverage")

    if logscale:
        ax1.set_xscale("log")
    ax1.set_xlabel("Test set coverage")
    ax1.set_ylabel("Objective value")
    # ax1.legend()

    ax2.plot(np.mean(np.vstack(df_standard['coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Violations']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")

    ax2.plot(np.mean(np.vstack(df_reshape['coverage_test']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Violations']),axis = 1)[beg2:end2], color="tab:orange", label=r"Reshaped set",alpha = 0.8)
    if dfs:
        for i in range(5):
            ax2.plot(np.mean(np.vstack(dfs[i+1][0]['coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(dfs[i+1][0]['Violations']),axis = 1)[beg1:end1], color="tab:blue", linestyle = "-")
            ax2.plot(np.mean(np.vstack(dfs[i+1][1]['coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(dfs[i+1][1]['Violations']),axis = 1)[beg2:end2], color = "tab:orange",linestyle = "-")
    # ax2.plot(np.arange(100)/100, 1 - np.arange(100)/100, color = "red")
    # ax2.set_ylim([-0.05,0.25])
    ax2.axvline(x = 0.8, color = "black",linestyle = ":", label = "0.8 Coverage")
    ax2.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
    ax2.set_ylabel("Prob. of cons. vio.")
    ax2.set_xlabel("Test set coverage")
    if logscale:
        ax2.set_xscale("log")
    # ax2.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    # ax2.legend()
    # lines_labels = [ax.get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels,loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2))
    plt.savefig(title+"_curves.pdf",bbox_inches='tight')
    plt.show()

def loss(t, x, y, Y, r,alpha, data,l=10000,quantile = 0.95, target = -0.1):
    sums = 0
    totsum = torch.tensor(0.)
    objs = 0
    Nsample = data.shape[0]
    for i in range(Nsample):
        sums += torch.sum(torch.maximum(x - (Y + torch.eye(J))@data[i] - y-alpha, torch.tensor(0.,requires_grad = True)))
        sums += torch.sum(torch.maximum(-y - Y@data[i] - alpha, torch.tensor(0.,requires_grad = True)))
        sums += torch.maximum((torch.tensor(c)-r)@x + r@y + r@Y@data[i] - t - alpha, torch.tensor(0.,requires_grad = True))

        objs += (torch.tensor(c)-r)@x + r@y + r@Y@data[i]
        
        newsums = torch.sum(torch.where(torch.maximum(-y - Y@data[i], torch.tensor(0.,requires_grad = True))>=0.0001,torch.tensor(1.), torch.tensor(0.)))
        newsums += torch.sum(torch.where(torch.maximum(x - (Y + torch.eye(J))@data[i] - y, torch.tensor(0.,requires_grad = True))>=0.0001,torch.tensor(1.), torch.tensor(0.)))
        newsums += torch.where(torch.maximum((torch.tensor(c)-r)@x + r@y + r@Y@data[i] - t, torch.tensor(0.,requires_grad = True))>=0.0001, torch.tensor(1.), torch.tensor(0.))
        if newsums >=1:
            totsum += torch.tensor(1.)
    sums = (sums/((1-quantile)*Nsample)) + alpha
    return t + torch.tensor(l)*(sums - torch.tensor(target)), objs/Nsample, totsum/Nsample, sums.detach().numpy()

u = lropt.UncertainParameter(J,
                        uncertainty_set=lropt.Ellipsoidal(p=2,
                                                    data=data, loss = loss))
# Formulate the Robust Problem
x = cp.Variable(J)
r = cp.Parameter(J)
y = cp.Variable(J)
Y = cp.Variable((J,J))
t = cp.Variable()
r.value = scenarios[0][0]
objective = cp.Minimize(t)
constraints = [(c-r)@x + r@y + r@Y@u <= t]
for i in range(J):
    constraints += [y[i] >= x[i] - (Y[i]+np.eye(J)[i])@u]
    constraints += [y[i] + Y[i]@u >= 0]
constraints += [x >= 0, cp.sum(x)<= q]
prob = lropt.RobustProblem(objective, constraints)

target = -0.05
test_p = 0.5
s = 5
train, test = train_test_split(data, test_size=int(data.shape[0]*test_p), random_state=s)
init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
init = np.eye(J)
init_bval = -init@np.mean(train, axis=0)
np.random.seed(1)
initn = init + 0.8*np.random.rand(J, J)
init_bvaln = -initn@np.mean(train, axis=0)

# Train A and b
result1 = prob.train(lr = 0.0001, step=100, momentum = 0.8, optimizer = "SGD", seed = s, init_A = init, init_b = init_bval, fixb = False,init_lam = 1, target_cvar = target, init_alpha = -0.005, test_percentage = test_p, scenarios = scenarios, num_scenarios = num_scenarios, step_y = 0.01,batch_percentage = 0.8)
df1 = result1.df
A_fin = result1.A
b_fin = result1.b

# Grid search epsilon
result4 = prob.grid(epslst = np.logspace(-2,2,100), init_A = init, init_b = init_bval, seed = s, init_alpha = 0., test_percentage =test_p,scenarios = scenarios, num_scenarios = num_scenarios)
dfgrid = result4.df

result5 = prob.grid(epslst = np.logspace(-2,2,100), init_A = A_fin, init_b = b_fin, seed = s, init_alpha = 0., test_percentage = test_p,scenarios = scenarios, num_scenarios = num_scenarios)
dfgrid2 = result5.df

plot_coverage_all(dfgrid,dfgrid2,None, "allocation",ind_1=(0,100),ind_2=(0,100), logscale = False)