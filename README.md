# lropt_experiments
This repository is by
[Irina Wang](https://sites.google.com/view/irina-wang),
Amit Soloman,
[Bart Van Parys](https://mitsloan.mit.edu/faculty/directory/bart-p-g-van-parys),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Learning Decision-Focused Uncertainty Sets in Robust Optimization](https://arxiv.org/abs/2305.19225)."
The experiments rely on our package [LROPT](https://github.com/stellatogrp/lropt).


If you find this repository helpful in your publications,
please consider citing our paper and package.

## Introduction
We propose a data-driven technique to automatically learn contextual uncertainty sets in robust optimization, optimizing both worst-case and average-case performance while also guaranteeing constraint satisfaction. 
Our method reshapes the uncertainty sets by minimizing the expected performance across a contextual family of problems, subject to conditional-value-at-risk constraints.
Our approach is very flexible, and can learn a wide variety of uncertainty sets while preserving tractability.
We solve the constrained learning problem using a stochastic augmented Lagrangian method that relies on differentiating the solutions of the robust optimization problems with respect to the parameters of the uncertainty set.
Due to the nonsmooth and nonconvex nature of the augmented Lagrangian function, we apply the nonsmooth conservative implicit function theorem to establish convergence to a critical point, which is a feasible solution of the constrained problem under mild assumptions.
Using empirical process theory, we show finite-sample probabilistic guarantees of constraint satisfaction for the resulting solutions.
Numerical experiments show that our method outperforms traditional approaches in robust and distributionally robust optimization in terms of out-of-sample performance and constraint satisfaction guarantees.

## Dependencies
From the root folder, install dependencies with
```
pip install -e ".[dev]"
```

## Instructions
### Running experiments
Experiments can be run from the root folder using the commands below. The code is parallelized, so it is recommended to run them on a computing cluster with 20+ cores.

Newsvendor Problem
```
python newsvendor/news.py
python newsvendor/gen_csv.py
```

Portfolio Optimization

For each of the following settings: $(m=30,N=2000)$, $(m=30,N=1000)$, $(m=10,N=2000)$, $(m=10,N=1000)$, run the following with the corresponding values of $m$ and $N$. We show an example for $m=30,N=2000$.

```
python portfolio/port.py --config-name=port_30_2000.yaml
python portfolio/port_ecro.py --config-name=port_ecro_30_2000.yaml
python portfolio/port_dro.py --config-name=port_dro_30_2000.yaml
python portfolio/port_dro_sep.py --config-name=port_dro_sep_30_2000.yaml
python portfolio/port_LCX_sep.py --config-name=lcx_30_2000.yaml
python gen_csv.py --m 30 --N 2000

```
Inventory Problem
```
python inventory/inv.py
python inventory/inv_dro.py
python inventory/inv_dro_sep.py
python inventory/gen_csv.py
```