import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import lropt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import sys
sys.path.append('..')
from utils import plot_iters, plot_coverage_all
warnings.filterwarnings("ignore")
from collections import Counter
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":18,
    "font.family": "serif"
})
colors = ["tab:blue", "tab:green", "tab:orange", 
          "blue", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "red"]


m = 50
np.random.seed(5)
distribution = np.random.dirichlet(np.exp(np.random.normal(size =m)))
np.random.seed(16)
poi = np.random.rand(m, 2)*2
purts = np.random.uniform(-1,1, (m, 2))*0.1
newpoints = poi + purts

from scipy.spatial import ConvexHull, convex_hull_plot_2d
rng = np.random.default_rng()
np.random.seed(18)
points = np.random.rand(5, 2)   # 30 random points in 2-D
hull = ConvexHull(points)
points1= np.random.rand(15, 2) + np.array((0.9,0.8))   # 30 random points in 2-D
hull1 = ConvexHull(points1)
points2= np.random.rand(15, 2) + np.array((0,1.1))   # 30 random points in 2-D
hull2 = ConvexHull(points2)
# plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1],'k-')
# plt.plot(points1[:,0], points1[:,1], 'o')
for simplex in hull1.simplices:
    plt.plot(points1[simplex, 0], points1[simplex, 1],'k-')
# plt.plot(points2[:,0], points2[:,1], 'o')
for simplex in hull2.simplices:
    plt.plot(points2[simplex, 0], points2[simplex, 1],'k-')
plt.plot(poi[:,0], poi[:,1], 'o')
plt.plot(newpoints[:,0], newpoints[:,1], 'o')
# plt.plot(x.value[0],x.value[1], 'o')
# plt.plot(x1.value[0],x1.value[1], 'o')
# plt.plot(x2.value[0],x2.value[1], 'o')
# plt.show()
# scenarios = {}
# num_scenarios = 5
# for scene in range(num_scenarios):
#     np.random.seed(scene+10)
#     points = np.random.rand(5, 2)   # 30 random points in 2-D
#     hull = ConvexHull(points)
#     np.random.seed(scene)
#     scenarios[scene] = {}
#     scenarios[scene][0] = hull.equations[:,:2]
#     scenarios[scene][1] = hull.equations[:,2]

y_data = []
num_scenarios = 5
for scene in range(num_scenarios):
    np.random.seed(scene+10)
    purts = np.random.uniform(-1,1, (m, 2))*0.1
    y_data.append(poi + purts)
y_data = np.array(y_data)


def gen_data(N,m,seed):
    data = np.zeros((N,m))
    np.random.seed(seed)
    for datint in range(N):
        sample = np.random.choice(np.arange(m), p=distribution,size = N)
        counter = Counter(sample)
        for i in range(m):
            data[datint][i] = counter[i]/N
    return data

def f_tch(tau, x,x1,x2,s, poi_y,u):
    # x is a tensor that represents the cp.Variable x.
    return tau

def g_tch(tau, x,x1,x2,s, poi_y,u):
    # x,y,u are tensors that represent the cp.Variable x and cp.Parameter y and 
    # The cp.Constant c is converted to a tensor
    return s@u.T - tau
    
def eval_tch(tau, x,x1,x2,s, poi_y,u):
    return s@u.T


seed = 0
for N in [50,80,100,300,500,1000,1500,2000,3000,5000]:
    seed += 1
    data = gen_data(N,m,seed)
    # formulate the ellipsoidal set
    D = np.vstack([-np.eye(m),np.ones(m), -np.ones(m)])
    d = np.hstack([np.zeros(m), 1, -1])

    u = lropt.UncertainParameter(m,
                                    uncertainty_set = lropt.Ellipsoidal(p=2, 
                                                                    rho=1., data =data, c= D, d = d))
    # formulate cvxpy variable
    tau = cp.Variable()
    s = cp.Variable(m)
    x = cp.Variable(2)
    x1 = cp.Variable(2)
    x2 = cp.Variable(2)
    poi_y = lropt.Parameter((m,2), data = y_data)
    poi_y.value = y_data[0]
    # formulate problem constants

    # formulate objective
    objective = cp.Minimize(tau)

    constraints = []
    # constraints += [normals@x + offsets <= 0]
    constraints += [hull.equations[:,:2]@x + hull.equations[:,2] <= 0]
    constraints += [hull1.equations[:,:2]@x1 + hull1.equations[:,2] <= 0]
    constraints += [hull2.equations[:,:2]@x2 + hull2.equations[:,2] <= 0]
    for k in range(m):
        constraints += [s[k] >= cp.norm(x - poi_y[k])]
        constraints += [s[k] >= cp.norm(x1 - poi_y[k])]
        constraints += [s[k] >= cp.norm(x2 - poi_y[k])]
    constraints += [s@u <= tau]
        
    # formulate Robust Problem
    prob = lropt.RobustProblem(objective, constraints,objective_torch=f_tch, constraints_torch=[g_tch], eval_torch=eval_tch)
    target = -0.01
    # solve
    test_p = 0.5
    train, test = train_test_split(data, test_size=int(data.shape[0]*test_p), random_state=5)
    init = np.real(sc.linalg.sqrtm(sc.linalg.pinv(np.cov(train.T))))
    # init = np.eye(m)
    init_bval = -init@np.mean(train, axis=0)
    np.random.seed(15)
    initn = init
    init_bvaln = -initn@np.mean(train, axis=0)
    #seed = 5,0
    result1 = prob.train(lr = 0.01, num_iter=3000, optimizer = "SGD", seed = 0, init_A = init, init_b = init_bval, init_lam = 0.5, init_mu = 0.01, mu_multiplier = 1.005, kappa = target, init_alpha = -0.0, test_percentage = test_p, lr_gamma = 0.2, lr_step_size = 300, position = True, random_init = True, num_random_init = 5)
    df1 = result1.df
    A_fin = result1.A
    b_fin = result1.b

    # Grid search epsilon
    result5 = prob.grid(epslst = np.linspace(0.0001,2, 200), init_A = A_fin, init_b = b_fin, seed = seed, init_alpha = 0., test_percentage = test_p)
    dfgrid2 = result5.df

    result4 = prob.grid(epslst = np.linspace(0.01, 3, 200), init_A = init, init_b = init_bval, seed = seed, init_alpha = 0., test_percentage =test_p)
    dfgrid = result4.df

    plot_coverage_all(dfgrid,dfgrid2,None, f"results/results1/facility(N,m)_{N,m}",ind_1=(0,800),ind_2=(0,800), logscale = False)

    plot_iters(df1,result1.df_test,f"results/results1/facility(N,m)_{N,m}", steps = 3000)
    
    dfgrid.to_csv(f"results/results1/gridmv_{N,m}.csv")
    dfgrid2.to_csv(f"results/results1/gridre_{N,m}.csv")
    result1.df_test.to_csv(f"results/results1/retrain_{N,m}.csv")


