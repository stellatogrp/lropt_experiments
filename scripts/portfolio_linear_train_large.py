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


def loss(t, x, p, alpha, data, l=10000, quantile=0.95, target=-0.1):
    sums = 0
    totsum = 0
    Nsample = data.shape[0]
    sums += torch.sum(torch.maximum(-data@x - t - alpha,
                      torch.tensor(0., requires_grad=True)))
    totsum += torch.sum(torch.where(torch.maximum(-data@x - t, torch.tensor(
        0., requires_grad=True)) >= 0.001, torch.tensor(1.), torch.tensor(0.)))
    sums1 = torch.mean(-data@x)
    sums = (sums/((1-quantile)*Nsample)) + alpha
    return t + 0.2*torch.norm(x-p, 1) + torch.tensor(l)*(sums - torch.tensor(target)), sums1 + 0.2*torch.norm(x-p, 1), totsum/Nsample, sums.detach().numpy()


def data_scaled(N, m, scale, seed):
    np.random.seed(seed)
    R = np.vstack([np.random.normal(
        0.9 + i*0.1*scale, np.sqrt((0.02**2+(i*0.1)**2)), N) for i in range(1, m+1)])
    return (R.transpose())


def data_modes(N, m, scales, seed):
    modes = len(scales)
    d = np.zeros((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,
          :] = data_scaled(weights, m, scales[i], seed)
    return d[0:N, :]


n = 15
N = 300
seed = 15
np.random.seed(seed)
dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
        20, 15, 10, 10, 10, 10, 10, 10])/10)[:n]
scenarios = {}
num_scenarios = 5
for scene in range(num_scenarios):
    np.random.seed(scene)
    scenarios[scene] = {}
    scenarios[scene][0] = np.reshape(np.random.dirichlet(dist, 1), (n,))
data = data_modes(600, n, [1, 2, 5], seed=15)

# Formulate uncertainty set
u = lropt.UncertainParameter(n,
                             uncertainty_set=lropt.Ellipsoidal(p=2,
                                                               data=data, loss=loss))
# Formulate the Robust Problem
x = cp.Variable(n)
t = cp.Variable()
p = cp.Parameter(n)
p.value = scenarios[0][0]
# p1.value = scenarios[0][1]
objective = cp.Minimize(t + 0.2*cp.norm(x - p, 1))

constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]

prob = lropt.RobustProblem(objective, constraints)
target = -0.05
test_p = 0.1
s = 5
train, test = train_test_split(data, test_size=int(
    data.shape[0]*test_p), random_state=s)
init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
init = np.eye(n)
init_bval = -init@np.mean(train, axis=0)
np.random.seed(15)
initn = np.random.rand(n, n)
init_bvaln = -initn@np.mean(train, axis=0)

# Train A and b
result1 = prob.train(lr=0.001, step=300, momentum=0.8, optimizer="SGD", seed=s, init_A=initn, init_b=init_bvaln, fixb=False, init_lam=1,
                     target_cvar=target, init_alpha=0., test_percentage=test_p, scenarios=scenarios, num_scenarios=num_scenarios, step_y=0.01)
df1 = result1.df
A_fin = result1.A
b_fin = result1.b

# Grid search epsilon
result4 = prob.grid(epslst=np.linspace(0.01, 3, 500), init_A=init, init_b=init_bval, seed=s,
                    init_alpha=0., test_percentage=test_p, scenarios=scenarios, num_scenarios=num_scenarios)
dfgrid = result4.df

result5 = prob.grid(epslst=np.linspace(0.01, 3, 500), init_A=A_fin, init_b=b_fin, seed=s,
                    init_alpha=0., test_percentage=test_p, scenarios=scenarios, num_scenarios=num_scenarios)
dfgrid2 = result5.df

plot_coverage_all(dfgrid, dfgrid2, None, "portlinear_large", ind_1=(
    0, 400), ind_2=(0, 400), logscale=False, zoom=False, legend=True, standard="Mean-Variance")
plot_iters(df1, "portlinear_large", steps=800, logscale=1)
