import cvxpy as cp
import scipy as sc
import numpy as np
import torch
import lropt
import sys
sys.path.append('..')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":12,
    "font.family": "serif"
})

T = 12
torch.set_default_dtype(torch.double)
K = 4
c = torch.tensor([(1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)])*0.1
p = torch.tensor(c*2.5)
h = torch.tensor(c*1.2)
d_star = np.array([1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)])
# d_star = 1000*np.ones(T)
proportion = 0.1
lhs = np.concatenate((np.eye(T), -np.eye(T)), axis=0)    
rhs_upper = (1 + proportion) * d_star
rhs_lower = (-1 + proportion) * d_star
rhs = np.hstack((rhs_upper, rhs_lower))
init_eps = 2
cov = 2000*np.eye(T)
Qmax = 3000
Vmin = -1000
Vmax = 12000
alpha1 = 0.1
alpha2 = 0.1
beta1 = 1
beta2 = 2
init_size = 100
demand_dist = torch.distributions.MultivariateNormal(torch.tensor(d_star),torch.tensor(cov))
cov = sc.linalg.sqrtm(cov)
torch.manual_seed(100)
sample_val = demand_dist.sample((100,))
# cov = sc.linalg.sqrtm(np.cov(sample_val.T))
init_dist = torch.distributions.Uniform(low = 0, high = 100)
eye_concat = np.eye(T+K-1)
eye_concat[T:,T:] = 0
Y_matref = np.ones((K,T+K-1))
Y_matref[:,T:] = 0
t_vals = torch.tensor(np.arange(T))
ones = np.ones(T)
zeros = np.zeros(T)
# et_vals = torch.tensor(np.array([np.concatenate([ones[:t+1], zeros[t+1:]]) for t in range(T)]))
def baseline_problem_aro(init_val, cval, pval, hval):
    d = lropt.UncertainParameter(T,uncertainty_set = lropt.Ellipsoidal(p=2,rho=init_eps,c = lhs, d = rhs,a = cov, b = d_star ))
    # d = d_star
    q = cp.Variable(T)
    y = cp.Variable(T)
    u = cp.Variable(T)
    z = cp.Variable(T-1)
    w = cp.Variable(T)
    u_var = cp.Variable((T,T))
    y_var = cp.Variable((T,T))
    q_var = cp.Variable((T,T))
    C = cp.Variable()
    x_init = init_val

    objective = C
    cons = [cval@(q + q_var@d) + cp.sum(y + y_var@d) + cp.sum(u + u_var@d) + cp.sum(z) - C]
    constraints = [0 <= q+ q_var@d,
                q+q_var@d <= Qmax]
    # constraints += [y_var == 0, u_var==0]
    for time in range(T):
        for time2 in range(time,T):
            constraints += [q_var[time,time2] == 0]
            # constraints += [y_var[time,time2] == 0]
            # constraints += [u_var[time,time2] == 0]

    for time in range(T):
        cons +=[Vmin -x_init - cp.sum((q+q_var@d)[:time+1]) + cp.sum(d[:(time+1)]) ]
        cons +=[x_init + cp.sum((q+q_var@d)[:(time+1)]) - cp.sum(d[:(time+1)])- Vmax]
        cons += [-(y + y_var@d)[time] + hval[time]*x_init + hval[time]*cp.sum((q+q_var@d)[:(time+1)]) - hval[time]*cp.sum(d[:(time+1)]) ]
        cons+= [-(y + y_var@d)[time] -pval[time]*x_init - pval[time]*cp.sum((q+q_var@d)[:(time+1)]) + pval[time]*cp.sum(d[:(time+1)]) ]
    constraints += [alpha1*(q+q_var@d -w) <= u + u_var@d ]
    constraints += [alpha2*(w-q-q_var@d)<=  u + u_var@d ]
    constraints += [ beta1*(w[1:] - w[:-1]) <= z]
    constraints += [beta2*(w[:-1] - w[1:]) <= z]
    constraints += [lropt.max_of_uncertain(cons)<=0]
    prob = lropt.RobustProblem(cp.Minimize(objective), constraints, eval_exp = cval@(q + q_var@d) + cp.sum(y + y_var@d) + cp.sum(u + u_var@d) + cp.sum(z))

    return prob, x_init, w,q,q_var,y,y_var,u,u_var,z

baseline_prob, x_init, w_baseline,q,q_var,y,y_var,u,u_var,z = baseline_problem_aro(init_val = 100, cval=c, pval=p, hval = h)
baseline_prob.solve()
objval, wval, qval, qmat, yval, ymat, uval, umat, zval = baseline_prob.objective.value, w_baseline.value, q.value, q_var.value, y.value, y_var.value, u.value, u_var.value, z.value
