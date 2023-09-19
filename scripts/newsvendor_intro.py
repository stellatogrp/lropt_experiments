from utils import plot_tradeoff, plot_iters, plot_contours_line, plot_coverage
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

# Formulate constants
n = 2
N = 1000
test_perc = 0.9
k = np.array([4., 5.])
p = np.array([5, 6.5])

# lagrangian function


def loss(t, x, y1, d1, y2, d2, k_tch, p_tch, alpha, data, l=5, quantile=0.95, target=1.):
    Nsample = data.shape[0]
    sums = torch.mean(torch.maximum(k_tch@x + y1 + data[:, 0]*d1 + y2 + data[:, 1]*d2 - t - alpha, torch.tensor(0., requires_grad=True)) + torch.maximum(
        torch.maximum(-data[:, 0]*(p_tch[0]), - x[0] *
                      (p_tch[0])) - y1 - data[:, 0]*d1 - alpha,
        torch.tensor(0., requires_grad=True)) + torch.maximum(
        torch.maximum(-data[:, 1]*(p_tch[1]), - x[1] *
                      (p_tch[1])) - y2 - data[:, 1]*d2 - alpha,
        torch.tensor(0., requires_grad=True)))
    # prob = torch.sum(torch.maximum(k_tch@x + y1 + data[:, 0]*d1 + y2 + data[:, 1]*d2 - t, torch.tensor(0., requires_grad=True)) + torch.maximum(
    #     torch.maximum(-data[:, 0]*p_tch[0], - p_tch[0]
    #                   * x[0]) - y1 - data[:, 0]*d1,
    #     torch.tensor(0., requires_grad=True)) +
    #     torch.maximum(torch.maximum(-data[:, 1]*p_tch[1], -p_tch[1]*x[1]) - y2 - data[:, 1]*d2,
    #                   torch.tensor(0., requires_grad=True)) >= torch.tensor(0.0001))/Nsample
    probn = torch.sum((torch.maximum(k_tch@x + torch.maximum(-data[:, 0]*p_tch[0], - p_tch[0]*x[0]) + torch.maximum(
        -data[:, 1]*p_tch[1], -p_tch[1]*x[1]) - t,  torch.tensor(0., requires_grad=True))) >= torch.tensor(0.0001))/Nsample
    sums = sums/(1-quantile) + alpha
    objective = torch.mean(k_tch@x + torch.maximum(-data[:, 0]*(p_tch[0]), - x[0]*(
        p_tch[0])) + torch.maximum(-data[:, 1]*(p_tch[1]), - x[1]*(p_tch[1])))
    return t + l*(sums - target), objective, probn, sums.detach().numpy()


def gen_demand_intro(N, seed):
    np.random.seed(seed)
    sig = np.array([[0.6, -0.4], [-0.3, 0.1]])
    mu = np.array((1.3, 1.3))
    norms = np.random.multivariate_normal(mu, sig, N)
    d_train = np.exp(norms)
    return d_train


# Generate instances
scenarios = {}
num_scenarios = 10
for scene in range(num_scenarios):
    np.random.seed(scene+3)
    scenarios[scene] = {}
    scenarios[scene][0] = k + np.random.normal(0, 0.3, n)
    scenarios[scene][1] = p + np.random.normal(0, 0.2, n)


# Formulate uncertainty set
experiment = 0
data = gen_demand_intro(N, seed=experiment)
u = lropt.UncertainParameter(n,
                             uncertainty_set=lropt.Ellipsoidal(p=2,
                                                               data=data, loss=loss))
# Formulate the Robust Problem
x_r = cp.Variable(n)
t = cp.Variable()
k = cp.Parameter(n)
p = cp.Parameter(n)
k.value = scenarios[0][0]
p.value = scenarios[0][1]
y1 = cp.Variable()
y2 = cp.Variable()
d1 = cp.Variable()
d2 = cp.Variable()

objective = cp.Minimize(t)
constraints = [k >= 0, p >= 0]
constraints += [k@x_r + y1 + d1 *
                np.array([1, 0])@u + y2 + d2*np.array([0, 1])@u <= t]
constraints += [cp.maximum(-p[0]*x_r[0], -p[0] *
                           np.array([1, 0])@u) <= y1 + d1*np.array([1, 0])@u]
constraints += [cp.maximum(-p[1]*x_r[1], -p[1] *
                           np.array([0, 1])@u) <= y2 + d2*np.array([0, 1])@u]
constraints += [x_r >= 0]

prob = lropt.RobustProblem(objective, constraints)
target = -0.1
s = 13

# setup intial A, b
train, test = train_test_split(data, test_size=int(
    data.shape[0]*test_perc), random_state=s)
init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
init_bval = -init@np.mean(train, axis=0)
np.random.seed(10)
initn = np.random.rand(n, n)
init_bvaln = -initn@np.mean(train, axis=0)

# Train shape
result1 = prob.train(lr=0.00007, step=300, momentum=0.5, optimizer="SGD", seed=s, init_A=initn, init_b=init_bvaln, fixb=False, init_lam=0.05, target_cvar=target,
                     init_alpha=-0.1, test_percentage=test_perc, save_iters=True, scenarios=scenarios, num_scenarios=num_scenarios, step_y=0.01, batch_percentage=1.0)
df1 = result1.df
A_fin = result1.A
b_fin = result1.b

# Grid search mean-var
result4 = prob.grid(epslst=np.logspace(-3, 2, 500), init_A=init, init_b=init_bval, seed=s,
                    init_alpha=-0., test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios)
dfgrid = result4.df

# Grid search reshaped
result5 = prob.grid(epslst=np.logspace(-3, 2, 500), init_A=A_fin, init_b=b_fin, seed=s,
                    init_alpha=-0., test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios)
dfgrid2 = result5.df

# plot objective vs epsilon
fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()
ax1.plot(np.mean(np.vstack(dfgrid['Eps']), axis=1)[:], np.mean(np.vstack(
    dfgrid['Test_val']), axis=1)[:], color="tab:blue", label=r"Mean-Var set obj.", zorder=0)
ax1.fill(np.append(np.quantile(np.vstack(dfgrid['Eps']), 0.25, axis=1), np.quantile(np.vstack(dfgrid['Eps']), 0.75, axis=1)[::-1]), np.append(
    np.quantile(np.vstack(dfgrid['Test_val']), 0.25, axis=1), np.quantile(np.vstack(dfgrid['Test_val']), 0.75, axis=1)[::-1]), color="tab:blue", alpha=0.2)

ax1.plot(np.mean(np.vstack(dfgrid2['Eps']), axis=1), np.mean(np.vstack(
    dfgrid2['Test_val']), axis=1), color="tab:orange", label="Reshaped set obj.", zorder=1)
ax1.fill(np.append(np.quantile(np.vstack(dfgrid2['Eps']), 0.25, axis=1), np.quantile(np.vstack(dfgrid2['Eps']), 0.75, axis=1)[::-1]), np.append(
    np.quantile(np.vstack(dfgrid2['Test_val']), 0.25, axis=1), np.quantile(np.vstack(dfgrid2['Test_val']), 0.75, axis=1)[::-1]), color="tab:orange", alpha=0.2)
ax1.set_ylabel("Objective value")
ax1.set_xlabel(r"$\epsilon$")
ax1.set_xscale("log")

ax2.plot(np.mean(np.vstack(dfgrid['Eps']), axis=1)[:], np.mean(np.vstack(dfgrid['Violations']), axis=1)[
         :], color="green", label=r"Mean-Var set $\hat{\eta}$", linestyle="-.", zorder=0)
ax2.fill(np.append(np.quantile(np.vstack(dfgrid['Eps']), 0.25, axis=1), np.quantile(np.vstack(dfgrid['Eps']), 0.75, axis=1)[::-1]), np.append(
    np.quantile(np.vstack(dfgrid['Violations']), 0.25, axis=1), np.quantile(np.vstack(dfgrid['Violations']), 0.75, axis=1)[::-1]), color="green", alpha=0.2)

ax2.plot(np.mean(np.vstack(dfgrid2['Eps']), axis=1), np.mean(np.vstack(
    dfgrid2['Violations']), axis=1), color="red", label=r"Reshaped set $\hat{\eta}$", linestyle="-.", zorder=1)
ax2.fill(np.append(np.quantile(np.vstack(dfgrid2['Eps']), 0.25, axis=1), np.quantile(np.vstack(dfgrid2['Eps']), 0.75, axis=1)[::-1]), np.append(
    np.quantile(np.vstack(dfgrid2['Violations']), 0.25, axis=1), np.quantile(np.vstack(dfgrid2['Violations']), 0.75, axis=1)[::-1]), color="red", alpha=0.2)
ax2.set_ylabel("Prob. of constr. vio. $(\hat{\eta})$")
fig.legend(bbox_to_anchor=(0.1, 0.9, 0, 0), loc="lower left",
           borderaxespad=0, ncol=4, fontsize=20)
plt.savefig("news_objective_vs_epsilon.pdf", bbox_inches='tight')

# plot statistics
plot_coverage(dfgrid, dfgrid2, "news", ind_1=(0, 500), ind_2=(0, 500))
plot_iters(df1, "news", steps=800, logscale=1)

# Get list of threshold epsilon
eps_list = np.logspace(-3, 2, 500)
prob_list = np.array([0., 0.01, 0.05, 0.1])
inds_standard = []
inds_reshaped = []
for i in prob_list:
    inds_standard.append(np.absolute(
        np.mean(np.vstack(dfgrid['Violations']), axis=1)-i).argmin())
    inds_reshaped.append(np.absolute(
        np.mean(np.vstack(dfgrid2['Violations'][:320]), axis=1)-i).argmin())
st_eps = eps_list[inds_standard[0]]
re_eps = eps_list[inds_reshaped[0]]

# plot objective vs violations
beg1, end1 = 240, 324
beg2, end2 = 260, 314
plt.figure(figsize=(15, 5))
plt.plot(np.mean(np.vstack(dfgrid['Violations']), axis=1)[beg1:end1], np.mean(np.vstack(
    dfgrid['Test_val']), axis=1)[beg1:end1], color="tab:blue", label=r"Mean-Var set", marker="v", zorder=0)
plt.fill(np.append(np.quantile(np.vstack(dfgrid['Violations']), 0.25, axis=1)[beg1:end1], np.quantile(np.vstack(dfgrid['Violations']), 0.75, axis=1)[beg1:end1][::-1]), np.append(
    np.quantile(np.vstack(dfgrid['Test_val']), 0.25, axis=1)[beg1:end1], np.quantile(np.vstack(dfgrid['Test_val']), 0.75, axis=1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)
for ind in range(4):
    plt.scatter(np.mean(np.vstack(dfgrid['Violations']), axis=1)[inds_standard[ind]], np.mean(np.vstack(
        dfgrid['Test_val']), axis=1)[inds_standard[ind]], color="tab:green", s=50, marker="v", zorder=10)
plt.plot(np.mean(np.vstack(dfgrid2['Violations']), axis=1)[beg2:end2], np.mean(np.vstack(
    dfgrid2['Test_val']), axis=1)[beg2:end2], color="tab:orange", label="Reshaped set", marker="^", zorder=1)
plt.fill(np.append(np.quantile(np.vstack(dfgrid2['Violations']), 0.25, axis=1)[beg2:end2], np.quantile(np.vstack(dfgrid2['Violations']), 0.75, axis=1)[beg2:end2][::-1]), np.append(
    np.quantile(np.vstack(dfgrid2['Test_val']), 0.25, axis=1)[beg2:end2], np.quantile(np.vstack(dfgrid2['Test_val']), 0.75, axis=1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
for ind in [0, 2, 1, 3]:
    plt.scatter(np.mean(np.vstack(dfgrid2['Violations']), axis=1)[inds_reshaped[ind]], np.mean(
        np.vstack(dfgrid2['Test_val']), axis=1)[inds_reshaped[ind]], color="black", s=50, marker="^")
plt.ylabel("Objective value")
# ax2.set_xlim([-1,20])
plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
# plt.xscale("log")
# plt.ylim([-2, 1])
plt.vlines(ymin=-3, ymax=1, x=prob_list[0], linestyles=":",
           color="tab:red", label="Reference $\hat{\eta}$")
for i in prob_list[1:]:
    plt.vlines(ymin=-3, ymax=1, x=i, linestyles=":", color="tab:red")
plt.legend()
plt.savefig("news_objective_vs_violations.pdf", bbox_inches='tight')

# solve for uncertainty set plots
theta = np.radians(0)
c, s = np.cos(theta), np.sin(theta)
rotation = np.array(((c, -s), (s, c)))
k = np.array([4., 5.])
p = np.array([5, 6.5])
x_opt_base = {}
x_opt_learned = {}
t_learned = {}
y1_learned = {}
y2_learned = {}
d1_learned = {}
d2_learned = {}
t_base = {}
y1_base = {}
y2_base = {}
d1_base = {}
d2_base = {}
for ind in range(4):
    x_opt_base[ind] = {}
    x_opt_learned[ind] = {}
    t_learned[ind] = {}
    t_base[ind] = {}
    y1_learned[ind] = {}
    y2_learned[ind] = {}
    d1_learned[ind] = {}
    d2_learned[ind] = {}
    y1_base[ind] = {}
    y2_base[ind] = {}
    d1_base[ind] = {}
    d2_base[ind] = {}
    for scene in range(num_scenarios):
        n = 2
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                       A=(1/eps_list[inds_standard[ind]])*init@rotation, b=(1/eps_list[inds_standard[ind]])*init@rotation@np.linalg.inv(init)@init_bval))
        # Formulate the Robust Problem
        x_r = cp.Variable(n)
        t = cp.Variable()
        k = cp.Parameter(n)
        p = cp.Parameter(n)
        k.value = scenarios[scene][0]
        p.value = scenarios[scene][1]
        y1 = cp.Variable()
        y2 = cp.Variable()
        d1 = cp.Variable()
        d2 = cp.Variable()

        objective = cp.Minimize(t)
        constraints = [k >= 0, p >= 0]
        constraints += [k@x_r + y1 + d1 *
                        np.array([1, 0])@u + y2 + d2*np.array([0, 1])@u <= t]
        constraints += [cp.maximum(-p[0]*x_r[0], -p[0] *
                                   np.array([1, 0])@u) <= y1 + d1*np.array([1, 0])@u]
        constraints += [cp.maximum(-p[1]*x_r[1], -p[1] *
                                   np.array([0, 1])@u) <= y2 + d2*np.array([0, 1])@u]
        constraints += [x_r >= 0]

        prob = lropt.RobustProblem(objective, constraints)
        prob.solve()
        x_opt_base[ind][scene] = x_r.value
        d1_base[ind][scene] = d1.value
        d2_base[ind][scene] = d2.value
        t_base[ind][scene] = t.value
        y1_base[ind][scene] = y1.value
        y2_base[ind][scene] = y2.value
        n = 2
        u = lropt.UncertainParameter(n,
                                     uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                       A=(1/eps_list[inds_reshaped[ind]])*A_fin@rotation, b=(1/eps_list[inds_reshaped[ind]])*A_fin@rotation@np.linalg.inv(A_fin)@b_fin))
        # Formulate the Robust Problem
        x_r = cp.Variable(n)
        t = cp.Variable()
        k = cp.Parameter(n)
        p = cp.Parameter(n)
        k.value = scenarios[scene][0]
        p.value = scenarios[scene][1]
        y1 = cp.Variable()
        y2 = cp.Variable()
        d1 = cp.Variable()
        d2 = cp.Variable()

        objective = cp.Minimize(t)
        constraints = [k >= 0, p >= 0]
        constraints += [k@x_r + y1 + d1 *
                        np.array([1, 0])@u + y2 + d2*np.array([0, 1])@u <= t]
        constraints += [cp.maximum(-p[0]*x_r[0], -p[0] *
                                   np.array([1, 0])@u) <= y1 + d1*np.array([1, 0])@u]
        constraints += [cp.maximum(-p[1]*x_r[1], -p[1] *
                                   np.array([0, 1])@u) <= y2 + d2*np.array([0, 1])@u]
        constraints += [x_r >= 0]
        prob = lropt.RobustProblem(objective, constraints)
        prob.solve()
        x_opt_learned[ind][scene] = x_r.value
        d1_learned[ind][scene] = d1.value
        d2_learned[ind][scene] = d2.value
        t_learned[ind][scene] = t.value
        y1_learned[ind][scene] = y1.value
        y2_learned[ind][scene] = y2.value
        x_opt_learned, x_opt_base, t_learned, t_base
A_fin, b_fin, init, init_bval

# Create the uncertainty sets and contours
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
# init_set = np.zeros((num_p,num_p))
    # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
    for i in range(num_p):
        for j in range(num_p):
            u_vec = [x[i, j], y[i, j]]
            for k_ind in range(K):
                # fin_set[ind][k_ind][i,j] = np.linalg.norm((1/eps_list[inds_reshaped[ind]])*A_fin[k_ind*n:(k_ind+1)*n, 0:n]@ u_vec + (1/eps_list[inds_reshaped[ind]])*b_fin)
                fin_set[ind][k_ind][i, j] = np.linalg.norm((1/eps_list[inds_reshaped[ind]])*A_fin[k_ind*n:(
                    k_ind+1)*n, 0:n] @ rotation@(u_vec + np.linalg.inv(A_fin[k_ind*n:(k_ind+1)*n, 0:n])@b_fin))

            for k_ind in range(K):
                # init_set[ind][k_ind][i,j] = np.linalg.norm((1/eps_list[inds_standard[ind]])*init[k_ind*n:(k_ind+1)*n, 0:n]@ u_vec  + (1/eps_list[inds_standard[ind]])*init_bval)
                init_set[ind][k_ind][i, j] = np.linalg.norm((1/eps_list[inds_standard[ind]])*init[k_ind*n:(
                    k_ind+1)*n, 0:n] @ rotation@(u_vec + np.linalg.inv(init[k_ind*n:(k_ind+1)*n, 0:n])@init_bval))

            for scene in range(num_scenarios):
                g_level_learned[ind][scene][i, j] = scenarios[scene][0]@x_opt_learned[ind][scene] + np.maximum(- scenarios[scene][1][0] * x_opt_learned[ind][scene][0], - scenarios[scene][1][0] * u_vec[0]) + np.maximum(
                    - scenarios[scene][1][1] * x_opt_learned[ind][scene][1], - scenarios[scene][1][1] * u_vec[1]) - t_learned[ind][scene]
                # scenarios[scene][0]@x_opt_learned[ind][scene] + y1_learned[ind][scene] + d1_learned[ind][scene]*u_vec[0] +  y2_learned[ind][scene] + d2_learned[ind][scene]*u_vec[1] - t_learned[ind][scene]

                g_level_base[ind][scene][i, j] = scenarios[scene][0]@x_opt_base[ind][scene] + np.maximum(- scenarios[scene][1][0] * x_opt_base[ind][scene][0], - scenarios[scene][1][0] * u_vec[0]) + np.maximum(
                    - scenarios[scene][1][1] * x_opt_base[ind][scene][1], - scenarios[scene][1][1] * u_vec[1]) - t_base[ind][scene]
                #scenarios[scene][0]@x_opt_base[ind][scene] + y1_base[ind][scene] + d1_base[ind][scene]*u_vec[0] +  y2_base[ind][scene] + d2_base[ind][scene]*u_vec[1]  - t_base[ind][scene]


def plot_contours_line(x, y, set, g_level, prob_list, num_scenarios, train, title, standard=True):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(16, 3.5), constrained_layout=True)
    ax_lst = [ax1, ax2, ax3, ax4]

    cur_ind = 0
    for axis in ax_lst:
        axis.set_title(r'$\hat{\eta}$' + ' = {}'.format(prob_list[cur_ind]))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        for scene in range(num_scenarios):
            axis.contour(x, y, g_level[cur_ind][scene], [0.3], colors=[
                         "tab:purple"], alpha=1, linestyles=["-"])
        axis.scatter(train[:, 0], train[:, 1],
                     color="white", edgecolor="black")
        axis.scatter(np.mean(train, axis=0)[0], np.mean(
            train, axis=0)[1], color=["tab:green"])
        for k_ind in range(1):
            axis.contour(x, y, set[cur_ind][k_ind], [1],
                         colors=["red"], linewidths=[2])
        cur_ind += 1
    if standard:
        post = "Mean-Variance"
    else:
        post = "Reshaped"
    fig.suptitle(post+" set", fontsize=30)
    plt.savefig(title+"_" + post + ".pdf", bbox_inches='tight')


plot_contours_line(x, y, init_set, g_level_base, prob_list,
                   num_scenarios, train, "news_intro", standard=True)
plot_contours_line(x, y, fin_set, g_level_learned, prob_list,
                   num_scenarios, train, "news_intro", standard=False)
