import cvxpy as cp 
import lropt as lropt
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import scipy as sc
import utils

def power(ind, results):
  plt.rcParams.update({
    "text.usetex": True,

    "font.size": 22,
    "font.family": "serif"
})
  # plt.figure(figsize = (9,4))
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
  ax1.plot([results.df_test["x_vals"][ind][t][5][0][0] for t in range(0,t_actual)], label = "$p_t$ commit")
  ax1.plot([results.df_test["x_vals"][ind][t][6][0][0] for t in range(1,t_actual+1)],label = r"$p_t$")
  ax1.plot(np.array([results.df_test["z_vals"][ind][t][1][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$p$ ch", linestyle = "-")
  ax1.plot(np.array([results.df_test["z_vals"][ind][t][0][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$p$ dis", linestyle = "-.")
  ax1.legend(bbox_to_anchor=(-0.1, 1))
  ax1.set_xlabel("time")

  ax2.plot(np.array([results.df_test["x_vals"][ind][t][6][0][0] for t in range(1,t_actual+1)]) - np.array([results.df_test["z_vals"][ind][t][0][0][0].detach().numpy() for t in range(t_actual) ]) + np.array([results.df_test["z_vals"][ind][t][1][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ grid", linestyle = "-.")
  ax2.plot(np.array([results.df_test["z_vals"][ind][t][2][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ store", linestyle = "-.")
  ax2.plot(np.array([results.df_test["z_vals"][ind][t][3][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ curt", linestyle = ":")
  ax2.plot([results.df_test["x_vals"][ind][t][2][0] for t in range(0,t_actual)], label = "energy")
  ax2.set_xlabel("time")
  ax2.legend(bbox_to_anchor=(1, 1))
  # ax2.legend(ncols = 1,bbox_to_anchor=(0.5, 1))
  profit =  np.array([results.df_test["x_vals"][ind][t][6][0][0] for t in range(1,t_actual+1)])@p_data[0][:t_actual]
  print("profit", profit)
  
  ax1.set_title("profit "+ str(round(-results.df_test['Test_val'][ind],3)))
  plt.show()


def test_iters(results):
  plt.figure(figsize = (6,2))
  plt.plot(results.df_test['Test_val'])
  plt.xlabel("Iteration")
  plt.ylabel("Eval (profit)")
  plt.show()
  plt.figure(figsize = (6,2))
  plt.plot(results.df_test['Violations_test'])
  plt.xlabel("Iteration")
  plt.ylabel("CVaR")
  plt.show()

def train_iters(results):
  plt.figure(figsize = (6,2))
  plt.plot(results.df['Train_val'],label="obj")
  plt.plot(results.df['Lagrangian_val'], label= "lagrangian")
  plt.xlabel("Iteration")
  plt.title("Lagrangian and training eval")
  plt.legend()
  plt.show()
  plt.figure(figsize = (6,2))
  plt.plot(results.df['Violations_train'])
  plt.xlabel("Iteration")
  plt.ylabel("CVaR")
  plt.show()

# wind forecast between 0 and 1, weibul distribution
# forecast errors with truncated normals, 10-15% relative standard deviation (of forecast x), truncate between [-x, P_W - x]
# price between 10, 80

T = 10 # time horizon
H = 5 # forecast range
N = 500
weights = np.array([0.1,0.1,0.2,0.3,0.3])

# what should these values be?
E_lower = 0
E_upper = .15 # kwh - 500 kilowatt, 2h charge/discharge (units in megawatt)
P_ES = .25 # 250 kilowatts 
P_W = 1 # max capacity of wind
gamma = 1

wind_fore_data = np.maximum(np.minimum(0.7*np.random.weibull(5,(N,T,H)),1),0)


def gen_error(N,seed,T,forecast):
    if seed!= 0:
        np.random.seed(seed)
    sig = 0.01*np.eye(T)
    points_list = []
    for i in range(N):
        mu = forecast[i]@weights
        newpoint = np.maximum(-mu,np.minimum(np.random.multivariate_normal(mu,sig) - mu,P_W - mu))
        points_list.append(newpoint)
    return np.vstack(points_list)

u_data = gen_error(N,0,T,wind_fore_data)
p_data = np.random.uniform(low=10,high=80,size=(N,T))

# decisions
# p_act = cp.Variable(T)
p_dis = cp.Variable(T)
p_dis_mat = cp.Variable((T,T))
p_ch = cp.Variable(T)
p_ch_mat = cp.Variable((T,T))
# w_grid = cp.Variable(T, nonneg = True)
w_store = cp.Variable(T)
w_store_mat = cp.Variable((T,T))
w_curt = cp.Variable(T)
w_curt_mat = cp.Variable((T,T))
# aux variable 
y_p = cp.Variable(T)
y_p_mat = cp.Variable((T,T))

# states
# p_commit = cp.Parameter(T)
# energy = cp.Parameter() # state of charge
# price = cp.Parameter(T)
# w_inj = cp.Parameter(T)
# p_net = cp.Parameter(T)
# price_fore = cp.Parameter(T)
wind_fore = lropt.ContextParameter((T,H),data=wind_fore_data)
# side_info =  cp.Parameter((T,H))

# wind forecast uncertainty, concatenated
u = lropt.UncertainParameter(T, uncertainty_set = lropt.Ellipsoidal(data=u_data))

energy = 0
np.random.seed(10)
p_commit_data = np.random.uniform(low=0.1,high=1,size=T)
price_fore = p_data[0][:T]
print(p_commit_data)

constraints = []
constraints += [p_ch <= P_ES, p_dis  <= P_ES]
constraints += [w_store <= p_ch]
constraints += [p_dis >=0]
constraints += [p_ch  >=0]
constraints += [w_curt >=0]
constraints += [w_store >=0]


for t in range(T):
  constraints += [E_lower <= energy + cp.sum((p_ch)[:t+1]) - cp.sum((p_dis)[:t+1])]
  constraints += [ energy + cp.sum((p_ch)[:t+1]) - cp.sum((p_dis)[:t+1]) <= E_upper]

  # p_t <= P_W +P_ES, p_t >= -P_ES
  
  constraints += [wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t] ) <= P_W + P_ES]
  constraints += [-P_ES <= wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t] )]

  # w_grid >= 0
  constraints += [wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t]) >= 0]

for t in range(T):
  # y_p = |p_t - p_t_commit|
  constraints += [y_p[t] >=wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t])-(p_ch[t]) - p_commit_data[t] ]
  constraints += [y_p[t] >= p_commit_data[t] - (wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t]))]

objective = cp.Minimize(gamma*cp.sum([y_p[t]  for t in range(T)]) - cp.sum([price_fore[t]*(wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t])) for t in range(T)]) + 2*cp.sum([p_ch[t] + p_dis[t] for t in range(T)]))

# eval_exp = gamma*cp.sum([y_p[t]  for t in range(T)]) - cp.sum([price_fore[t]*(wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t])-(p_ch[t] )) for t in range(T) ]) + 2*cp.sum([p_ch[t] + p_dis[t] for t in range(T)])

eval_exp = -cp.sum([price_fore[t]*(wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t])-(p_ch[t] )) for t in range(T)])

prob = lropt.RobustProblem(objective, constraints,eval_exp = eval_exp)

s = 5
np.random.seed(s)
test_p = 0.5
test_indices = np.random.choice(N,int(test_p*N), replace=False)
train_indices = [i for i in range(N) if i not in test_indices]
train = np.array([u_data[i] for i in train_indices])
test = np.array([u_data[i] for i in test_indices])
initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)

# Train A and b
# init_bias = np.hstack([initn.flatten(),mults_mean_bias])
# init_weight = np.vstack([np.zeros((675,125)),mults_mean_weight])
# predictor = lropt.LinearPredictor()

from lropt import Trainer
trainer = Trainer(prob)
trainer_settings = lropt.TrainerSettings()
trainer_settings.lr=0.0001
trainer_settings.train_size = False
trainer_settings.num_iter=101
trainer_settings.optimizer="SGD"
trainer_settings.seed=5
trainer_settings.init_A=initn
trainer_settings.init_b=init_bvaln
trainer_settings.init_lam=0.1
trainer_settings.init_mu=0.1
trainer_settings.init_rho=1
trainer_settings.save_history = True
trainer_settings.lr_step_size = 50
trainer_settings.lr_gamma = 0.5
trainer_settings.test_percentage = test_p
trainer_settings.test_frequency = 5
trainer_settings.random_init = False
trainer_settings.parallel = False
trainer_settings.eta=0.3
trainer_settings.contextual = True
trainer_settings.predictor = lropt.LinearPredictor(predict_mean=True)
trainer_settings.kappa = 0.0
# trainer_settings.predictor = predictor
# trainer_settings.init_weight = init_weight
# trainer_settings.init_bias = init_bias
result_ro = trainer.train(settings=trainer_settings)
df = result_ro.df
A_fin = result_ro.A
b_fin = result_ro.b



p_hat_ro = wind_fore@weights

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
ax1.plot(p_commit_data, label = r"$p$ commit")
ax1.plot(wind_fore_data[0]@weights + u_data[0] - w_curt.value -w_store.value + p_dis.value - p_ch.value ,label = r"$p_t$")
ax1.plot(p_ch.value, label = "$p$ ch")
ax1.plot(p_dis.value, label = "$p$ dis")
ax1.set_xlabel("time")
ax1.legend(bbox_to_anchor=(-0.1, 1))


ax2.plot( wind_fore_data[0]@weights + u_data[0] - w_curt.value -w_store.value ,label = r"$w$ grid", linestyle = "-.")
ax2.plot(w_store.value ,label = r"$w$ store", linestyle = "-.")
ax2.plot(w_curt.value ,label = r"$w$ curt", linestyle = ":")
ax2.plot(np.cumsum(p_ch.value) - np.cumsum(p_dis.value), label = "energy")
ax2.set_xlabel("time")
ax2.legend(bbox_to_anchor=(1, 1))
profit = (wind_fore_data[0]@weights + u_data[0] - w_curt.value -w_store.value + p_dis.value - p_ch.value)@price_fore
print("profit", result_ro.df_test["Test_val"][int(np.floor(trainer_settings.num_iter/trainer_settings.test_frequency))])
plt.show()