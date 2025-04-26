import cvxpy as cp 
import lropt as lropt
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import scipy as sc
def power(ind, results):
  plt.rcParams.update({
    "text.usetex": True,

    "font.size": 22,
    "font.family": "serif"
})
  # plt.figure(figsize = (9,4))
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
  ax1.plot([results.df_test["x_vals"][ind][t][6][0][0] for t in range(0,t_actual)], label = "$p_t$ commit")
  ax1.plot([results.df_test["x_vals"][ind][t][7][0][0] for t in range(1,t_actual+1)],label = r"$p_t$")
  ax1.plot(np.array([results.df_test["z_vals"][ind][t][1][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$p$ ch", linestyle = "-")
  ax1.plot(np.array([results.df_test["z_vals"][ind][t][0][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$p$ dis", linestyle = "-.")
  ax1.legend(bbox_to_anchor=(-0.1, 1))
  ax1.set_xlabel("time")

  ax2.plot(np.array([results.df_test["x_vals"][ind][t][7][0][0] for t in range(1,t_actual+1)]) - np.array([results.df_test["z_vals"][ind][t][0][0][0].detach().numpy() for t in range(t_actual) ]) + np.array([results.df_test["z_vals"][ind][t][1][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ grid", linestyle = "-.")
  ax2.plot(np.array([results.df_test["z_vals"][ind][t][2][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ store", linestyle = "-.")
  ax2.plot(np.array([results.df_test["z_vals"][ind][t][3][0][0].detach().numpy() for t in range(t_actual)]) ,label = r"$w_t$ curt", linestyle = ":")
  ax2.plot([results.df_test["x_vals"][ind][t][3][0] for t in range(0,t_actual)], label = "energy")
  ax2.set_xlabel("time")
  ax2.legend(bbox_to_anchor=(1, 1))
  # ax2.legend(ncols = 1,bbox_to_anchor=(0.5, 1))
  profit =  np.array([results.df_test["x_vals"][ind][t][7][0][0] for t in range(1,t_actual+1)])@p_data[0][:t_actual]
  print("profit", profit)
  
  ax1.set_title("test obj "+ str(round(-results.df_test['Test_val'][ind],3)))
  plt.show()

def test_iters(results,test_frequency):
  plt.figure(figsize = (6,2))
  plt.plot(np.arange(len(results.df_test['Test_val']))*test_frequency, results.df_test['Test_val'])
  plt.xlabel("Iteration")
  plt.ylabel("Eval")
  plt.show()
  plt.figure(figsize = (6,2))
  plt.plot(np.arange(len(results.df_test['Test_val']))*test_frequency, results.df_test['Violations_test'])
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

def gen_wind_data(N,T,H,seed=0,mult = 0.6):
    np.random.seed(seed)
    wind_fore_data = np.zeros((N,T,H))
    init_wind = np.maximum(np.minimum(0.7*np.random.weibull(5,(N)),1),0)
    wind_fore_data[:,0,0] = init_wind
    winds = [init_wind]
    prev_wind_init = init_wind.copy()
    prev_wind = prev_wind_init.copy()
    for j in range(1,H):
        new_wind = mult*prev_wind + (1-mult)*np.maximum(np.minimum(0.7*np.random.weibull(5,(N)),1),0)
        wind_fore_data[:,0,j] = new_wind
        prev_wind = new_wind.copy()
    for t in range(1,T):
        prev_wind =  mult*prev_wind_init + (1-mult)*np.maximum(np.minimum(0.7*np.random.weibull(5,(N)),1),0)
        wind_fore_data[:,t,0] = prev_wind
        prev_wind_init = prev_wind.copy()
        for j in range(H):
            new_wind =  mult*prev_wind + (1-mult)*np.maximum(np.minimum(0.7*np.random.weibull(5,(N)),1),0)
            wind_fore_data[:,t,j] = new_wind
            prev_wind = new_wind.copy()
    return wind_fore_data
wind_fore_data = gen_wind_data(N,T+1,H,0,mult=0.6)

# wind forecast between 0 and 1, weibul distribution
# forecast errors with truncated normals, 10-15% relative standard deviation (of forecast x), truncate between [-x, P_W - x]
# price between 10, 80

def gen_error(N,seed,T,forecast):
    if seed!= 0:
        np.random.seed(seed)
    sig = 0.01*np.eye(T)
    errors = np.zeros((N,T))
    for i in range(N):
        mu = forecast[i]@weights
        prev = np.maximum(-mu[0],np.minimum(np.random.normal(mu[0],0.01) - mu[0],P_W - mu[0]))
        errors[i,0] = prev
        for time in range(1,T):
            newpoint = 0.9*prev + 0.1*np.maximum(-mu[time],np.minimum(np.random.normal(mu[time],0.01) - mu[time],P_W - mu[time]))
            errors[i,time] = newpoint
            prev = newpoint
    return errors

u_data = gen_error(N,0,T+1,wind_fore_data)[:,1:]
p_data = np.random.uniform(low=10,high=80,size=(N,T))
wind_fore_data = wind_fore_data[:,1:,:]


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
obj_var = cp.Variable()

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


t_actual = 10
K = 3
T = t_actual + K # time horizon
H = 5 # forecast range
N = 500
weights = np.array([0.1,0.1,0.2,0.3,0.3])

# what should these values be?
E_lower = 0
E_upper = .15
# E_upper = .5 # kwh - 500 kilowatt, 2h charge/discharge (units in megawatt)
P_ES = .25 # 250 kilowatts 
P_W = 1 # max capacity of wind
gamma = 1

wind_fore_data = gen_wind_data(N,T+1,H)
# wind_fore_data = torch.cat((torch.tensor(wind_fore_data_orig),torch.zeros((N,K,H))),dim=1)

u_data = gen_error(N,0,T+1,wind_fore_data)
# u_data = torch.cat((torch.tensor(u_data_orig),torch.zeros((N,K))),dim=1)
# p_data = np.random.uniform(low=10,high=80,size=(N,T))
p_cat = torch.cat((torch.tensor(p_data),torch.zeros((N,K))),dim=1)
p_commit_data_cat = torch.cat((torch.tensor(p_commit_data),torch.zeros((K))))
# mults_mean_weight, mults_mean_bias = gen_weights_bias(wind_fore_data,u_data)

initn = sc.linalg.sqrtm(np.cov(u_data[:,1:].T))
init_bvaln = np.mean(u_data[:,1:], axis=0)
# init_bvaln_cat = torch.cat((torch.tensor(init_bvaln),torch.zeros(K)))

# decisions
# p_act = cp.Variable(T)

p_dis = cp.Variable(K)
p_ch = cp.Variable(K)
w_store = cp.Variable(K)
w_curt = cp.Variable(K)
# aux variable 
y_p = cp.Variable(K)
price_fore_var = cp.Variable(K)
obj_var = cp.Variable()


# states
wind_mu = cp.Parameter(K)
wind_mu_prev = cp.Parameter(1)
wind_fore = cp.Parameter((K,H))
energy = cp.Parameter(1)
p_commit = cp.Parameter(K)
p_hat = cp.Parameter(K)
wind_fore_all = cp.Parameter((T,H))
t_param = cp.Parameter(1)
price_fore = cp.Parameter(K)
fore_error_all = cp.Parameter(T)


# wind forecast uncertainty
u = lropt.UncertainParameter(K, uncertainty_set = lropt.Ellipsoidal(data=np.zeros((1,K))))

# p_commit = 0.1*np.ones(T)
# price_fore = p_data[0]


constraints = []
constraints += [p_ch <= P_ES, p_dis  <= P_ES]
constraints += [w_store <= p_ch]
constraints += [p_dis >=0]
constraints += [p_ch  >=0]
constraints += [w_curt >=0]
constraints += [w_store >=0]
constraints += [wind_mu >= -200, t_param >=0, p_hat >= -200, wind_fore_all >= -200, fore_error_all >= -200,wind_mu_prev >= -200]

for t in range(K):
  constraints += [E_lower <= energy + cp.sum((p_ch)[:t+1]) - cp.sum((p_dis)[:t+1])]
  constraints += [ energy + cp.sum((p_ch)[:t+1]) - cp.sum((p_dis)[:t+1]) <= E_upper]

  # p_t <= P_W +P_ES, p_t >= -P_ES
  
  constraints += [wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t] ) <= P_W + P_ES]
  constraints += [-P_ES <= wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t] )]

  # w_grid >= 0
  constraints += [wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t]) >= 0]

for t in range(K):
  # y_p = |p_t - p_t_commit|
  constraints += [y_p[t] >= wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t])-(p_ch[t]) - p_commit[t] ]
  constraints += [y_p[t] >= p_commit[t] - (wind_fore[t]@weights+ u[t] - (w_curt[t]) -(w_store[t])+(p_dis[t] )-(p_ch[t]))]
# constraints += [y_p[K-1] == 0]

constraints += [price_fore_var == price_fore]
constraints += [gamma*cp.sum([y_p[t]  for t in range(K)]) - cp.sum([(price_fore_var[t]*wind_fore[t]@weights+ price_fore_var[t]*u[t] - (price_fore[t]*w_curt[t]) -(price_fore[t]*w_store[t])+(price_fore[t]*p_dis[t] )-(price_fore[t]*p_ch[t])) for t in range(K)])+ 2*cp.sum([p_ch[t] + p_dis[t] for t in range(K)])<= obj_var]
objective = cp.Minimize(obj_var)

prob = lropt.RobustProblem(objective, constraints)
trainer = lropt.Trainer(prob)
policy = trainer.create_cvxpylayer(parameters = [wind_mu, wind_mu_prev,wind_fore,energy,price_fore,t_param,p_commit,p_hat,wind_fore_all,fore_error_all], variables = [p_dis,p_ch,w_store,w_curt,y_p,price_fore_var,obj_var])


class PowerSimulator(lropt.Simulator):
    def simulate(self, x, u):
        p_dis,p_ch,w_store,w_curt,y_p,pv,_ = u
        wind_mu,wind_mu_prev,wind_fore,energy,price,tval,p_commit,p_hat,wind_fore_all,fore_error_all = x
        batch_size = p_dis.shape[0]
        energy_new = energy + p_ch[:,0:1] - p_dis[:,0:1]
        # fore_error_new = torch.tensor(gen_error(batch_size,0,K,wind_fore.numpy()))
        t = int(tval[0])
        fore_error_new = fore_error_all[:,t:t+K]
        p_hat_new = torch.matmul(wind_fore,torch.tensor(weights)) + fore_error_new - w_curt - w_store - p_ch + p_dis
        # p_commit[:,:-1] = p_hat_new[:,1:]
        t = t+1
        p_commit = (p_commit_data_cat[t:t+K]).repeat(batch_size,1)
        wind_mu_new = torch.tensor(init_bvaln[t:t+K]).repeat(batch_size,1)
        wind_fore_new = wind_fore_all[:,t:t+K]
        new_price = (p_cat[0][t:t+K]).repeat(batch_size,1)
        context = [wind_mu_new,fore_error_new[:,0:1],wind_fore_new,energy_new,new_price,tval+1,p_commit,p_hat_new,wind_fore_all,fore_error_all]
        
        return context

    def stage_cost_eval(self,x,u):
        p_dis,p_ch,w_store,w_curt,y_p ,pv,_= u
        wind_mu, _,wind_fore,energy,price,tval,p_commit,p_hat,wind_fore_all,_ = x
        t = int(tval[0])-1
        return (gamma*y_p[:,0] - p_data[0][t]*p_hat[:,0] + 2*p_dis[:,0] + 2*p_ch[:,0]).mean()


    def stage_cost(self,x,u):
        p_dis,p_ch,w_store,w_curt,y_p,pv,_ = u
        wind_mu, _, wind_fore,energy,price,tval,p_commit,p_hat,wind_fore_all,_ = x
        t = int(tval[0])-1
        return (gamma*y_p[:,0] - p_data[0][t]*p_hat[:,0] + 2*p_dis[:,0] + 2*p_ch[:,0]).mean()


    def constraint_cost(self,x,u,alpha):
        wind_mu, _, wind_fore,energy,price,tval,p_commit,p_hat,wind_fore_all,_ = x
        batch_size = p_hat.shape[0]
        cvar_term =(1/0.3)*torch.max(torch.max(-P_ES -p_hat[:,0],
                        p_hat[:,0] - P_W - P_ES) - alpha,
                        torch.zeros(batch_size)) + alpha
        return ((cvar_term + 0.00)).mean()

    def init_state(self,batch_size, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        t = 0
        energy = torch.zeros((batch_size,1))
        wind_fore_all = torch.tensor(gen_wind_data(batch_size,T+1,H,seed=seed,mult = 0.6))
        wind_fore_all_needed = wind_fore_all[:,1:,:]
        # wind_fore_all = torch.cat((torch.tensor(wind_fore_all),torch.zeros((batch_size,K,H))),dim=1)
        fore_error_all =  torch.tensor(gen_error(batch_size,0,T+1,wind_fore_all.numpy()))
        fore_error_all_needed = fore_error_all[:,1:]
        wind_fore = wind_fore_all_needed[:,t:t+K]
        p_commit = (p_commit_data_cat[t:t+K]).repeat(batch_size,1)
        p_hat = torch.zeros((batch_size,K),dtype=torch.double)
        t_new = (torch.tensor(t,dtype=torch.double)).repeat(batch_size,1)
        wind_mu = torch.tensor(init_bvaln[t:t+K]).repeat(batch_size,1)
        wind_mu_prev = fore_error_all[:,0:1]
        price =(p_cat[0][t:t+K]).repeat(batch_size,1)
        context = [wind_mu,wind_mu_prev,wind_fore,energy,price,t_new,p_commit,p_hat,wind_fore_all_needed,fore_error_all_needed]
        return context

    def prob_constr_violation(self, x, u, **kwargs):
        p_dis,p_ch,w_store,w_curt,y_p,pv,_ = u
        wind_mu, _,wind_fore,energy,price,tval,p_commit,p_hat,wind_fore_all,_ = x
        return (((torch.max(torch.stack((-P_ES -p_hat[:,0], p_hat[:,0] - P_W - P_ES,  -p_hat[:,0] +p_dis[:,0] - p_ch[:,0]),dim=1),dim=1)[0])>= torch.tensor(0.)).float()).mean()/T

simulator = PowerSimulator()

epochs = 101
batch_size = 5
test_batch_size = 5
test_frequency = 5
lr = 0.01
# init_x0 = simulator.init_state(seed = 0, batch_size = 100)
init_a = initn[:K,:K]
init_b = init_bvaln[:K]
x_endind = K*H + K + 1
init_weights = torch.zeros((K*K+K,x_endind))
init_weights[K*K:,:K] = torch.eye(K)
trainer_settings = lropt.TrainerSettings()
trainer_settings.set(simulator=simulator, multistage=True, policy=policy, time_horizon=t_actual,
                    num_iter=epochs, batch_size=batch_size, init_rho=0.5, seed=0,
                    init_A=init_a, init_b=init_b, optimizer="SGD", lr=lr, momentum=0,
                    init_alpha=-0.00, scheduler=True, lr_step_size=50, lr_gamma=0.8,
                    contextual=True, test_batch_size=test_batch_size, x_endind=x_endind, init_lam=0.5, init_mu=1.5, max_iter_line_search = 20,
                    init_weight= None,
                    mu_multiplier=1.01,  test_frequency = test_frequency, parallel = False,line_search_threshold = 1,init_uncertain_param = u_data[:,1:K+1],predictor=lropt.LinearPredictor(predict_mean = True))
results = trainer.train(settings=trainer_settings)
