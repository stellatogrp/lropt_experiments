import numpy as np
from scipy.stats import bernoulli

def findlast_sort(val, sort_list, TOL=1e-10, start=1):
    for i in range(start, len(sort_list)):
        if abs(val - sort_list[i]) > TOL:
            break
    return i - 1

def singlepass(zetas, zetahats):
    zetas.sort()
    zetahats.sort()
    vstar = np.mean(zetas) - zetas[0]
    vb = np.mean(zetahats) - zetas[0]
    Gamma = vstar - vb
    N = len(zetas)
    pbar = 1.0
    hat_indx = 1
    hat_indx_ = 0
    for k in range(1, len(zetas)):
        vstar += (zetas[k-1] - zetas[k]) * (N-k) / N
        hat_indx = findlast_sort(zetas[k-1], zetahats, start=hat_indx_ + 1)
        pbar -= (hat_indx - hat_indx_) / N
        hat_indx_ = hat_indx
        vb += (zetas[k-1] - zetas[k]) * pbar
        Gamma = max(Gamma, vstar - vb)
    return Gamma

def f2(boot_sample, data, numSamples, a, sgns):
    Gamma = 0.0
    for _ in range(numSamples):
        a = randL1(a, sgns)
        Gamma_ = singlepass(np.dot(data, a), np.dot(boot_sample, a))
        Gamma = max(Gamma, Gamma_)
    return Gamma

def randL1(a, sgns):
    a = np.random.rand(len(a))
    a /= np.sum(a)
    sgns = bernoulli.rvs(0.5, size=len(sgns))
    a *= 2 * sgns - 1
    return a

def calc_ab_thresh(data, alpha, numBoots, numSamples):
    N, d = data.shape
    a = np.zeros(d)
    sgns = np.zeros(d, dtype=int)
    return boot(data, f2, 1-alpha, numBoots, data, numSamples, a, sgns)

def boot(data, fun, prob, numBoots, *f_args):
    N = data.shape[0]
    out = np.zeros(numBoots)
    indices = np.arange(N)
    for i in range(numBoots):
        sampled_indices = np.random.choice(indices, size=N, replace=True)
        out[i] = fun(data[sampled_indices, :], *f_args)
    return np.quantile(out, prob)
