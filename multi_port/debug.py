import cvxpy as cp
import scipy as sc
import numpy as np
import torch
import lropt
import sys
import torch.nn.functional as f
sys.path.append('..')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":24,
    "font.family": "serif"
})

ETFS = ['AGG', 'VTI', 'VNQ', 'XLF', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLE', 'IBB', 'ITA']
N_ASSETS = len(ETFS)
HZN = 24
VAL = 2443
TEST = VAL * 3
KAPPA = np.full(N_ASSETS, 0.001)
SHORT = np.full(N_ASSETS, 0.001)
GAMMA = 15.0

M1 = torch.tensor(2.0)
M2 = torch.tensor(1.00)
len(ETFS)
MU = np.load('multi_port/data/markowitz_mu.npy')
COV = np.load('multi_port/data/markowitz_sigma.npy')
COV_SQRT = sc.linalg.sqrtm(COV)

mulog = torch.tensor(np.load('multi_port/data/markowitz_mu_log.npy'))
covlog = torch.tensor(np.load('multi_port/data/markowitz_cov_log.npy'))

def init_holdings(cov_sqrt, mu, gamma):
    N = N_ASSETS
    ht = cp.Variable(N)
    risk = gamma * cp.sum_squares(cov_sqrt @ ht)
    returns = mu.T @ ht
    objective = returns - risk - SHORT.T @ cp.neg(ht)
                                                                                
    constraints = [                                                             
        cp.sum(ht) == 1,
    ]                                                                            
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    return ht.value

H0 = torch.tensor(init_holdings(COV_SQRT, MU, GAMMA))

init_size = 100
kappa_tch = torch.from_numpy(KAPPA)
logreturn1p_dist = torch.distributions.MultivariateNormal(
    mulog, covlog)

N = N_ASSETS
htall = lropt.Parameter(N+1, data= np.zeros((init_size,N+1)))   
mu = lropt.UncertainParameter(N,uncertainty_set = lropt.Ellipsoidal(p=2,rho=1, data = np.zeros((init_size,N))))
ht = htall[:N]
                                                                            
utall = cp.Variable(N+1)
htp = cp.Variable(N) 
ut, tau = utall[:N], utall[N:]      
                                                                            
objective = tau
                                                                            
transaction_cost = KAPPA.T @ cp.abs(ut)
shorting_cost = SHORT.T @ cp.neg(htp)
constraints = [
    -mu.T @ htp <= tau,
    cp.sum(ut) + transaction_cost + shorting_cost <= 0,
    htp == ht + ut,
]                                                                            
prob = lropt.RobustProblem(cp.Minimize(objective), constraints)
trainer = lropt.Trainer(prob)
policy = trainer.create_cvxpylayer(variables = [utall])

class PortSimulator(lropt.Simulator):

  def simulate(self, x, u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    ret = torch.exp(logreturn1p_dist.sample((batch_size,)))
    newx = ret * (x[:,:N] + u[:,:N])
    # newxnorm = f.normalize(newx,p=1,dim=1)
    xsums = torch.sum(newx,axis=1).view((batch_size,1))
    newxnorm = newx/xsums
    x = torch.cat([newxnorm,xsums],axis=1)
    return x

  def stage_cost(self,x,u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    xval,sval = x[:,:N], x[:,N:]
    return -self.utility_fn(sval)
  
  def utility_fn(self,x, m1=M1, m2=M2):
    # return x
    return torch.min(m1*(x - 1), m2*(x - 1))

  # def stage_cost(self,x,u):
  #   assert x.shape[0] == u.shape[0]
  #   batch_size = x.shape[0]
  #   r_batch = r_th.repeat(batch_size, 1, 1)
  #   tau_batch = tau_th.repeat(batch_size, 1, 1)
  #   h, p, dh = x[:,:n], x[:, n:n+k], x[:, n+k:]
  #   s_vec = torch.cat([p, tau_batch, -r_batch], 1).double()
  #   S = torch.bmm(s_vec.transpose(1, 2), u)
  #   H = alpha * h + beta * (h ** 2)
  #   return torch.sum(S, 1) + torch.sum(H, 1)

  def constraint_cost(self,x,u,alpha):
    eta = 0.05
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    xval,sval = x[:,:N], x[:,N:]
    uval, tau = u[:,:N], u[:,N:]
    cvar_term =(1/eta)*(torch.max(torch.max(-sval - tau,axis=1)[0] - alpha,torch.zeros(batch_size))[0]) + alpha
    return 0.0*cvar_term
  
  # def constraint_cost(self,x,u,alpha):
  #   eta = 0.05
  #   assert x.shape[0] == u.shape[0]
  #   batch_size = x.shape[0]
  #   h, p, dh = x[:,:n], x[:, n:n+k], x[:, n+k:]
  #   cvar_term =(1/eta)*(torch.max(torch.max(u[:,retail_links,:] - dh,axis=1)[0] - alpha,torch.zeros(batch_size))[0]) + alpha
  #   return 0.01*cvar_term

  def init_state(self,batch_size, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    H0 = torch.tensor(init_holdings(COV_SQRT, MU, GAMMA))
    ret = torch.exp(logreturn1p_dist.sample((batch_size,)))
    newH = ret*H0
    # z =  torch.normal(0,0.1,size=(batch_size, N)).double()
    # newH = H0+z
    means = torch.sum(newH,axis=1).view((batch_size,1))
    newH = newH/means
    # newH = f.normalize(newH,p=1,dim=1)
    zer = torch.zeros(batch_size, 1).double()
    x_batch = torch.cat([newH,zer],axis=1)
    return x_batch
simulator = PortSimulator()
# Perform training
time_horizon = 24
epochs = 50
batch_size = 10
lr = 0.001
# init_x = simulator.init_state(seed = 0, batch_size = 100)
# init_h = init_x[:,:N]
init_a = COV_SQRT
init_b = MU
val_costs, val_costs_constr, \
  paramvals, x_base, u_base = trainer.multistage_train(simulator, 
                                                       policy = policy, 
                         time_horizon = time_horizon, epochs = epochs, 
                         batch_size = batch_size, init_eps=1, seed=0,
                          init_a = init_a, init_b = init_b,
                          optimizer = "SGD",lr= lr, momentum = 0, init_alpha = 0.0, scheduler = False)