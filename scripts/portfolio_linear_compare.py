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


# generate instances
n = 2
N = 300
seed = 15
np.random.seed(seed)
dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
        20, 15, 10, 10, 10, 10, 10, 10])/100)[:n]
scenarios = {}
num_scenarios = 1
for scene in range(num_scenarios):
    np.random.seed(scene)
    scenarios[scene] = {}
    scenarios[scene][0] = np.reshape(np.random.dirichlet(dist, 1), (n,))

# generate data


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
objective = cp.Minimize(t + 0.2*cp.norm(x - p, 1))

constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]

prob = lropt.RobustProblem(objective, constraints)
target = -0.05
test_p = 0.1
s = 5
train, test = train_test_split(data, test_size=int(
    data.shape[0]*test_p), random_state=s)
# init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
init = np.eye(n)
init_bval = -init@np.mean(train, axis=0)
np.random.seed(0)
A_fin = np.random.rand(n, n)
b_fin = -A_fin@np.mean(train, axis=0)

# Grid search epsilon
result4 = prob.grid(epslst=np.linspace(0.01, 3, 500), init_A=init, init_b=init_bval, seed=s,
                    init_alpha=0., test_percentage=test_p, scenarios=scenarios, num_scenarios=num_scenarios)
dfgrid = result4.df

# construct and plot uncertainty sets
eps_list = np.linspace(0.01, 3, 500)
prob_list = np.linspace(0, 0.5, 10)
inds_standard = []
for i in prob_list:
    inds_standard.append(np.absolute(
        np.mean(np.vstack(dfgrid['Violations']), axis=1)-i).argmin())
    
inds_reshaped = []
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
for ind in range(10):
    x_opt_base[ind] = {}
    x_opt_learned[ind] = {}
    t_learned[ind] = {}
    t_base[ind] = {}
    for scene in range(num_scenarios):
        n = 2
        Amat = (1/eps_list[inds_standard[ind]])*init
        bvec = (1/eps_list[inds_standard[ind]])*init_bval
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                       A=Amat, b=bvec))
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
        rho2 = (bvec@np.linalg.inv(Amat.T)@x.value + np.linalg.norm(np.linalg.inv(Amat.T)@x.value, 2) -
                b_fin@np.linalg.inv(A_fin.T)@x.value)/np.linalg.norm(np.linalg.inv(A_fin.T)@x.value, 2)
        inds_reshaped.append(rho2)
        t_st.append(t.value)
        x_opt_base[ind][scene] = x.value
        t_base[ind][scene] = t.value

        n = 2
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2, rho=rho2, A=A_fin, b=b_fin))
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
for ind in range(10):
    fin_set[ind] = {}
    init_set[ind] = {}
    for k_ind in range(K):
        fin_set[ind][k_ind] = np.zeros((num_p, num_p))
        init_set[ind][k_ind] = np.zeros((num_p, num_p))
g_level_learned = {}
g_level_base = {}
for ind in range(10):
    g_level_learned[ind] = {}
    g_level_base[ind] = {}
    for scene in range(num_scenarios):
        g_level_learned[ind][scene] = np.zeros((num_p, num_p))
        g_level_base[ind][scene] = np.zeros((num_p, num_p))
    for i in range(num_p):
        for j in range(num_p):
            u_vec = [x[i, j], y[i, j]]
            for k_ind in range(K):
                fin_set[ind][k_ind][i, j] = np.linalg.norm(
                    (1/inds_reshaped[ind])*A_fin[k_ind*n:(k_ind+1)*n, 0:n] @ u_vec + (1/inds_reshaped[ind])*b_fin)

            for k_ind in range(K):
                init_set[ind][k_ind][i, j] = np.linalg.norm((1/eps_list[inds_standard[ind]])*init[k_ind*n:(
                    k_ind+1)*n, 0:n] @ u_vec + (1/eps_list[inds_standard[ind]])*init_bval)

            for scene in range(num_scenarios):
                g_level_learned[ind][scene][i, j] = - \
                    x_opt_learned[ind][scene]@u_vec - t_learned[ind][scene]
                g_level_base[ind][scene][i, j] = - \
                    x_opt_base[ind][scene]@u_vec - t_base[ind][scene]


def plot_contours_line(x, y, set, g_level, prob_list, num_scenarios, train, title, standard=True):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(16, 3.5), constrained_layout=True)
    ax_lst = [ax1, ax2, ax3, ax4]

    cur_ind = 0
    for axis in ax_lst:
        axis.set_title(r'$\hat{\eta}$' + ' = {}'.format(round(prob_list[cur_ind],3)))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        for scene in range(num_scenarios):
            axis.contour(x, y, g_level[cur_ind][scene], [0], colors=[
                         "tab:purple"], alpha=1, linestyles=["-"])
        axis.scatter(test[:, 0], test[:, 1], color="white", edgecolor="black")
        axis.scatter(np.mean(test, axis=0)[0], np.mean(
            train, axis=0)[1], color=["tab:green"])
        for k_ind in range(1):
            axis.contour(x, y, set[cur_ind][k_ind], [1],
                         colors=["red"], linewidths=[2])
        cur_ind += 3
    if standard:
        post = "Identity"
    else:
        post = "Reshaped"
    fig.suptitle(post+" set", fontsize=30)
    plt.savefig(title+"_" + post + ".pdf", bbox_inches='tight')


# plot_contours_line(x, y, init_set, g_level_base, prob_list,
#                    num_scenarios, train, "portlinear", standard=True, standard_name="Identity")
plot_contours_line(x, y, fin_set, g_level_learned, prob_list,
                   num_scenarios, train, "portlinear_random", standard=False)


plt.figure(figsize=(15, 5))
plt.plot(prob_val_st, test_val_st, marker="^",
         label="Identity set out-of-sample")
plt.plot(prob_val_st, t_st, label="Identity set in-sample")
plt.plot(prob_val_re, test_val_re, marker="v",
         label="Random set out-of-sample")
plt.plot(prob_val_re, t_re, label="Random set in-sample")
plt.ylabel("Objective value")
plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
plt.legend()
plt.savefig("portlinear_compare.pdf", bbox_inches='tight')
