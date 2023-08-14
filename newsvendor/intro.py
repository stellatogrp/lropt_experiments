import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.size": 24, "font.family": "serif"})


# Formulate constants
n = 2
N = 1000
test_perc = 0.9


def gen_demand_intro(N, seed):
    np.random.seed(seed)
    sig = np.array([[0.6, -0.4], [-0.3, 0.1]])
    mu = np.array((0.9, 0.7))
    norms = np.random.multivariate_normal(mu, sig, N)
    d_train = np.exp(norms)
    return d_train


# Generate data
data = gen_demand_intro(N, seed=8)
