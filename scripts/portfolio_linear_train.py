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


n = 2
N = 300
seed = 15
np.random.seed(seed)
dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
        20, 15, 10, 10, 10, 10, 10, 10])/100)[:n]
scenarios = {}
num_scenarios = 5
for scene in range(num_scenarios):
    np.random.seed(scene)
    scenarios[scene] = {}
    scenarios[scene][0] = np.reshape(np.random.dirichlet(dist, 1), (n,))


def gen_demand_intro(N, seed):
    np.random.seed(seed)
    sig = np.array([[0.6, -0.4], [-0.3, 0.1]])
    mu = np.array((0.3, 0.3))
    norms = np.random.multivariate_normal(mu, sig, N)
    d_train = np.exp(norms)
    return d_train


data = gen_demand_intro(600, seed=5)

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

plot_coverage_all(dfgrid, dfgrid2, None, "portlinear", ind_1=(
    0, 400), ind_2=(0, 400), logscale=False, zoom=False, legend=True, standard="Identity")
plot_iters(df1, "portlinear", steps=800, logscale=1)

# construct and plot uncertainty sets
eps_list = np.linspace(0.01, 3, 500)
prob_list = np.array([0., 0.01, 0.05, 0.1])
inds_standard = []
inds_reshaped = []
for i in prob_list:
    inds_standard.append(np.absolute(
        np.mean(np.vstack(dfgrid['Violations']), axis=1)-i).argmin())
    inds_reshaped.append(np.absolute(
        np.mean(np.vstack(dfgrid2['Violations']), axis=1)-i).argmin())

inds = []
x_opt_base = {}
x_opt_learned = {}
t_learned = {}
t_base = {}
test_val_st = []
prob_val_st = []
test_val_re = []
prob_val_re = []
t_re = []
t_st = []
for ind in range(4):
    x_opt_base[ind] = {}
    x_opt_learned[ind] = {}
    t_learned[ind] = {}
    t_base[ind] = {}
    for scene in range(num_scenarios):
        n = 2
        Amat = (1/eps_list[inds_reshaped[ind]])*A_fin
        bvec = (1/eps_list[inds_reshaped[ind]])*b_fin
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2, A=Amat, b=bvec))
        # Formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()
        p = cp.Parameter(n)
        p.value = scenarios[scene][0]
        objective = cp.Minimize(t + 0.2*cp.norm(x - p, 1))

        constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]
        # constraints += [x == x_opt_base[ind][scene]]

        prob = lropt.RobustProblem(objective, constraints)
        prob.solve()
        test_val_re.append(np.mean(-test@x.value) + 0.2 *
                           np.linalg.norm(x.value-p.value, 1))
        prob_val_re.append(
            np.sum(np.where(np.maximum(-test@x.value - t.value, 0) >= 0.001, 1, 0))/60)
        x_opt_learned[ind][scene] = x.value
        t_re.append(t.value)
        t_learned[ind][scene] = t.value
        rho2 = (bvec@np.linalg.inv(Amat.T)@x.value + np.linalg.norm(np.linalg.inv(Amat.T)@x.value, 2) -
                init_bval@np.linalg.inv(init.T)@x.value)/np.linalg.norm(np.linalg.inv(init.T)@x.value, 2)
        inds.append(rho2)

        n = 2
        Amat = (1/eps_list[inds_standard[ind]])*init
        bvec = (1/eps_list[inds_standard[ind]])*init_bval
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2, A=Amat, b=bvec))
        # Formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()
        p = cp.Parameter(n)
        p.value = scenarios[scene][0]
        objective = cp.Minimize(t + 0.2*cp.norm(x - p, 1))

        constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]

        prob = lropt.RobustProblem(objective, constraints)
        prob.solve()
        test_val_st.append(np.mean(-test@x.value) + 0.2 *
                           np.linalg.norm(x.value-p.value, 1))
        prob_val_st.append(
            np.sum(np.where(np.maximum(-test@x.value - t.value, 0) >= 0.001, 1, 0))/60)
        t_st.append(t.value)
        x_opt_base[ind][scene] = x.value
        t_base[ind][scene] = t.value

K = 1
num_p = 50
offset = 2
x_min, x_max = np.min(train[:, 0]) - offset, np.max(train[:, 0]) + offset
y_min, y_max = np.min(train[:, 1]) - offset, np.max(train[:, 1]) + offset
X = np.linspace(x_min, x_max, num_p)
Y = np.linspace(y_min, y_max, num_p)
x, y = np.meshgrid(X, Y)
# Z values as a matrix
fin_set = {}
init_set = {}
for ind in range(4):
    fin_set[ind] = {}
    init_set[ind] = {}
    for k_ind in range(K):
        fin_set[ind][k_ind] = np.zeros((num_p, num_p))
        init_set[ind][k_ind] = np.zeros((num_p, num_p))
g_level_learned = {}
g_level_base = {}
for ind in range(4):
    g_level_learned[ind] = {}
    g_level_base[ind] = {}
    for scene in range(num_scenarios):
        g_level_learned[ind][scene] = np.zeros((num_p, num_p))
        g_level_base[ind][scene] = np.zeros((num_p, num_p))
    for i in range(num_p):
        for j in range(num_p):
            u_vec = [x[i, j], y[i, j]]
            for k_ind in range(K):
                fin_set[ind][k_ind][i, j] = np.linalg.norm((1/eps_list[inds_reshaped[ind]])*A_fin[k_ind*n:(
                    k_ind+1)*n, 0:n] @ u_vec + (1/eps_list[inds_reshaped[ind]])*b_fin)
                # fin_set[ind][k_ind][i,j] = np.linalg.norm((1/inds_reshaped[ind])*A_fin[k_ind*n:(k_ind+1)*n, 0:n]@ u_vec + (1/inds_reshaped[ind])*b_fin)

            for k_ind in range(K):
                init_set[ind][k_ind][i, j] = np.linalg.norm((1/eps_list[inds_standard[ind]])*init[k_ind*n:(
                    k_ind+1)*n, 0:n] @ u_vec + (1/eps_list[inds_standard[ind]])*init_bval)

            for scene in range(num_scenarios):
                g_level_learned[ind][scene][i, j] = - \
                    x_opt_learned[ind][scene]@u_vec - t_learned[ind][scene]
                g_level_base[ind][scene][i, j] = - \
                    x_opt_base[ind][scene]@u_vec - t_base[ind][scene]

plot_contours_line(x, y, init_set, g_level_base, prob_list,
                   num_scenarios, train, "portlinear", standard=True, standard_name="Identity")
plot_contours_line(x, y, fin_set, g_level_learned, prob_list,
                   num_scenarios, train, "portlinear", standard=False)
