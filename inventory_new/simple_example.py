import lropt as cvxro
import cvxpy as cp
n = 5
unc_data = 0
a = 0
Bu=0
d= 0
P = 0
q = 0


x = cp.Variable(n)
u = cvxro.UncertainParameter(n,
                            uncertainty_set = cvxro.Ellipsoidal(data = unc_data))
constraints = [(a + Bu)@x <= d]
objective = cp.Minimize(cp.quad_form(P,x)+ q@x)
problem = cvxro.RobustProblem(objective, constraints)
Trainer = cvxro.Trainer(problem)
result = Trainer.train()
