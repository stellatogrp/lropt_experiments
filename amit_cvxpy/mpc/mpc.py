import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
# import lropt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import scipy.sparse as scs
import sys
# import ot
sys.path.append('..')
# from utils import plot_iters, plot_coverage_all
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":18,
    "font.family": "serif"
})
colors = ["tab:blue", "tab:green", "tab:orange", 
          "blue", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "red"]


M = 5
l = 2
n = 8
m = 3
A = np.reshape(np.array([0.19,0.35,0.03,0.00,0.71,0.14,0.01,0.00,0.35,0.22,0.35,0.04,0.14,0.71,0.14,0.01,0.03,0.35,0.23,0.39,0.01,0.14,0.71,0.14,0.00,0.04,0.39,0.58,0.00,0.01,0.14,0.85,-1.28,0.44,0.12,0.01,0.19,0.35,0.03,0.00,0.44,-1.15,0.45,0.13,0.35,0.22,0.35,0.04,0.12,0.45,-1.15,0.57,0.03,0.35,0.23,0.39,0.01,0.13,0.57,-0.71,0.00,0.04,0.39,0.58]),(8,8))
B = np.reshape(np.array([0.39,0.00,-0.04,-0.39,0.04,-0.42,-0.04,0.39,-0.04,-0.00,-0.42,-0.00,0.57,0.01,-0.14,-0.58,0.13,-0.71,-0.13,0.57,-0.14,-0.01,-0.71,-0.01]),(8,3))
D = np.array([[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1]]).T

def setup(A, B, D, M=M, l=l, n=n,m=m):
  Q_ = {}
  for i in range(M-1):
    Q_[i] = np.zeros((n,n))
  Q_[4] = np.bmat([[np.eye(4),np.zeros((4,4))],[np.zeros((4,4)),np.zeros((4,4))]])

  R_ = {}
  for i in range(M):
    R_[i] = 10e-6*np.eye(m)

  Q = sc.linalg.block_diag(*[Q_[i] for i in range(M)])
  Q = scs.csr_matrix(Q)

  R = sc.linalg.block_diag(*[R_[i] for i in range(M)])
  R = scs.csr_matrix(R)

  F = np.vstack([np.linalg.matrix_power(A,i) for i in range(1,M+1)])

  G = {}
  for i in range(M):
    firsthalf = np.hstack([np.linalg.matrix_power(A,j)@B for j in np.flip(range(i+1))])
    if M-i-1 > 0:
      secondhalf = np.hstack([np.zeros((n,m))]*(M-i-1))
    if i + 1 < M:
      G[i] = np.hstack([firsthalf,secondhalf])
    else: 
      G[i] = firsthalf
  G = np.vstack([G[i] for i in range(M)])

  H = {}
  for i in range(M):
    firsthalf = np.hstack([np.linalg.matrix_power(A,j)@D for j in np.flip(range(i+1))])
    if M-i-1 > 0:
      secondhalf = np.hstack([np.zeros((n,l))]*(M-i-1))
    if i + 1 < M:
      H[i] = np.hstack([firsthalf,secondhalf])
    else: 
      H[i] = firsthalf
  H = np.vstack([H[i] for i in range(M)])
  return Q, R, F, G, H

Q, R, F, G, H = setup(A, B, D)
N = 1500
np.random.seed(10)
data = np.random.normal(size=(N,l*M))
testdat = np.random.normal(size=(N,l*M))
y_bar = 1.8
u_bar = 1.8


def construct_nominal(epsilon,S,C,K, M=M,n=n,m=m,l=l):
    gamma = cp.Variable(m*M)
    theta = cp.Variable((M*m,M*l))
    eta = cp.Variable()
    lam = cp.Variable()
    t = cp.Variable(K)
    w = cp.Parameter(K)
    dat = cp.Parameter((K,l*M))
    rho = cp.Parameter()
    objective = cp.quad_form(F@np.zeros(n) + G@(gamma + theta@np.mean(data,axis=0))+ H@np.mean(data,axis=0),Q) + cp.quad_form(gamma + theta@np.mean(data,axis=0),R)
    constraints = [eta + lam*rho + w@t <= 0]
    for i in range(M):
        constraints += [theta[i*m:(i+1)*m, i*l:]==0]
        
    for i in range(M): 
        for j in range(m):
            constraints +=[cp.norm((1/epsilon)*S[j]@(theta[(i*m):(i+1)*m]))<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - u_bar + S[j]@gamma[(i*m):(i+1)*m] + S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - u_bar - S[j]@gamma[(i*m):(i+1)*m] - S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]
    for i in range(M): 
        for j in range(4):
            constraints +=[cp.norm((1/epsilon)*C[j]@((H+G@theta)[(i*n):(i+1)*n]))<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - y_bar + C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) + C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - y_bar - C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) - C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]] 
    constraints += [t >=0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, gamma, theta, eta, lam, t, w, dat, rho

def eval_nominal(testdat,gamma, theta, eta, epsilon, S,C, M=M):
    evalval = 0
    for k in range(testdat.shape[0]): 
        evalval += np.maximum(np.maximum(np.max([np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)])) - eta.value,0)
    evalval = eta.value + (evalval/testdat.shape[0])/epsilon

    eval1 = 0
    for k in range(testdat.shape[0]): 
        eval1 += np.maximum(np.max([np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)]))
    eval1 = eval1/testdat.shape[0]
    return evalval, eval1

def construct_varycvar(epsilon,S,C,K, M=M,n=n,m=m,l=l):
    gamma = cp.Variable(m*M)
    theta = cp.Variable((M*m,M*l))
    eta = cp.Variable(M)
    lam = cp.Variable()
    t = cp.Variable(K)
    w = cp.Parameter(K)
    dat = cp.Parameter((K,l*M))
    rho = cp.Parameter()
    objective = cp.quad_form(F@np.zeros(n) + G@(gamma + theta@np.mean(data,axis=0))+ H@np.mean(data,axis=0),Q) + cp.quad_form(gamma + theta@np.mean(data,axis=0),R)
    constraints = [lam*rho + w@t <= 0]
    for i in range(M):
        constraints += [theta[i*m:(i+1)*m, i*l:]==0]
      
    for i in range(M): 
        for j in range(m):
            constraints +=[cp.norm((1/epsilon[i])*S[j]@(theta[(i*m):(i+1)*m]))<= lam]
            for k in range(K):
                constraints += [eta[i] +(1/epsilon[i])*(-eta[i]  - u_bar + S[j]@gamma[(i*m):(i+1)*m] + S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]
                constraints += [eta[i] +(1/epsilon[i])*(-eta[i] - u_bar - S[j]@gamma[(i*m):(i+1)*m] - S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]

    for i in range(M): 
        for j in range(4):
            constraints +=[cp.norm((1/epsilon[i])*C[j]@((H+G@theta)[(i*n):(i+1)*n]))<= lam]
            for k in range(K):
                constraints += [eta[i] + (1/epsilon[i])*(-eta[i]- y_bar + C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) + C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]]
                constraints += [eta[i] + (1/epsilon[i])*(-eta[i]- y_bar - C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) - C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]] 

    for i in range(M):
        constraints += [t >= eta[i]]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, gamma, theta, eta, lam, t, w, dat, rho

def eval_cvar(testdat,gamma, theta, eta, epsilon, S,C, M=M):
    evalval = 0
    for k in range(testdat.shape[0]): 
        evalval += np.maximum(np.max([eta[i].value + np.maximum((np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar -eta[i].value),0)/epsilon[i] for i in range(M)]), np.max([eta[i].value + np.maximum((np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar - eta[i].value),0)/epsilon[i] for i in range(M)]))
    evalval = evalval/testdat.shape[0]
    eval1 = 0
    for k in range(testdat.shape[0]): 
        eval1 += np.maximum(np.max([np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)]))
    eval1 = eval1/testdat.shape[0]

    return evalval, eval1

def construct_varyf(epsilon,S,C,K, M=M,n=n,m=m,l=l):
    gamma = cp.Variable(m*M)
    theta = cp.Variable((M*m,M*l))
    eta = cp.Variable()
    lam = cp.Variable()
    t = cp.Variable(K)
    w = cp.Parameter(K)
    dat = cp.Parameter((K,l*M))
    rho = cp.Parameter()
    objective = cp.quad_form(F@np.zeros(n) + G@(gamma + theta@np.mean(data,axis=0))+ H@np.mean(data,axis=0),Q) + cp.quad_form(gamma + theta@np.mean(data,axis=0),R)
    constraints = [eta + lam*rho + w@t <= 0]
    for i in range(M):
        constraints += [theta[i*m:(i+1)*m, i*l:]==0]
      
    for i in range(M): 
        for j in range(m):
            constraints +=[cp.norm((1/epsilon)*S[i][j]@(theta[(i*m):(i+1)*m]))<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - u_bar + S[i][j]@gamma[(i*m):(i+1)*m] + S[i][j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - u_bar - S[i][j]@gamma[(i*m):(i+1)*m] - S[i][j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]

    for i in range(M): 
        for j in range(4):
            constraints +=[cp.norm((1/epsilon)*C[j]@((H+G@theta)[(i*n):(i+1)*n]))<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - y_bar + C[i][j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) + C[i][j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - y_bar - C[i][j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) - C[i][j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]] 
    constraints += [t>=0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, gamma, theta, eta, lam, t, w, dat, rho

def eval_f(M,testdat,gamma, theta, eta, epsilon, S,C):
    evalval = 0
    for k in range(testdat.shape[0]): 
        evalval += np.maximum(np.maximum(np.max([np.linalg.norm(C[i]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S[i]@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)])) - eta.value,0)
    evalval = eta.value + (evalval/testdat.shape[0])/epsilon

    eval1 = 0
    for k in range(testdat.shape[0]): 
        eval1 += np.maximum(np.max([np.linalg.norm(C[i]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S[i]@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)]))
    eval1 = eval1/testdat.shape[0]

    return evalval, eval1

def construct_varydist(epsilon,S,C,K, D_bar_inv, M=M,n=n,m=m,l=l):
    gamma = cp.Variable(m*M)
    theta = cp.Variable((M*m,M*l))
    eta = cp.Variable()
    lam = cp.Variable()
    t = cp.Variable(K)
    w = cp.Parameter(K)
    dat = cp.Parameter((K,l*M))
    rho = cp.Parameter()
    objective = cp.quad_form(F@np.zeros(n) + G@(gamma + theta@np.mean(data,axis=0))+ H@np.mean(data,axis=0),Q) + cp.quad_form(gamma + theta@np.mean(data,axis=0),R)
    constraints = [eta + lam*rho + w@t <= 0]
    for i in range(M):
        constraints += [theta[i*m:(i+1)*m, i*l:]==0]
      
    for i in range(M): 
        for j in range(m):
            constraints +=[cp.norm((1/epsilon)*S[j]@(theta[(i*m):(i+1)*m])@D_bar_inv)<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - u_bar + S[j]@gamma[(i*m):(i+1)*m] + S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - u_bar - S[j]@gamma[(i*m):(i+1)*m] - S[j]@(theta[(i*m):(i+1)*m]@dat[k])) <= t[k]]

    for i in range(M): 
        for j in range(4):
            constraints +=[cp.norm((1/epsilon)*C[j]@((H+G@theta)[(i*n):(i+1)*n])@D_bar_inv)<= lam]
            for k in range(K):
                constraints += [(1/epsilon)*(-eta - y_bar + C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) + C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]]
                constraints += [(1/epsilon)*(-eta - y_bar - C[j]@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma)[(i*n):(i+1)*n]) - C[j]@(((H+G@theta)[(i*n):(i+1)*n])@dat[k])) <= t[k]] 

    constraints += [t >=0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, gamma, theta, eta, lam, t, w, dat, rho

def eval_dist(testdat,gamma, theta, eta, epsilon, S,C, M=M):
    evalval = 0
    for k in range(testdat.shape[0]): 
        evalval += np.maximum(np.maximum(np.max([np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)])) - eta.value,0)
    evalval = eta.value + (evalval/testdat.shape[0])/epsilon

    eval1 = 0
    for k in range(testdat.shape[0]): 
        eval1 += np.maximum(np.max([np.linalg.norm(C@(F[(i*n):(i+1)*n]@np.zeros(n) + (G@gamma.value)[(i*n):(i+1)*n] + (H+G@theta.value)[(i*n):(i+1)*n]@testdat[k]),np.inf) - y_bar for i in range(M)]), np.max([np.linalg.norm(S@(gamma.value[(i*m):(i+1)*m] + theta.value[(i*m):(i+1)*m]@testdat[k]), np.inf) - u_bar for i in range(M)]))
    eval1 = eval1/testdat.shape[0]

    return evalval, eval1


