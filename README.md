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
We propose a data-driven technique to automatically learn the uncertainty sets in robust optimization.
Our method reshapes the uncertainty sets by minimizing the expected performance across a family of problems subject to guaranteeing constraint satisfaction. 
Our approach is very flexible and can learn a wide variety of uncertainty sets while preserving tractability.
We solve the constrained learning problem using a stochastic augmented Lagrangian method that relies on differentiating the solutions of the robust optimization problems with respect to the parameters of the uncertainty set.
Due to the nonsmooth and nonconvex nature of the augmented Lagrangian function, we apply the nonsmooth conservative implicit function theorem to establish convergence to a critical point which is a locally optimal solution of the constrained problem under mild assumptions.
Using empirical process theory, we show finite-sample probabilistic guarantees of constraint satisfaction for the resulting solutions.
Numerical experiments show that our method outperforms traditional approaches in robust and distributionally robust optimization in terms of out-of-sample performance and constraint satisfaction guarantees.

## Dependencies
Install dependencies with
```
pip install git+https://github.com/stellatogrp/lropt.git@develop#egg=lropt[dev]
```

## Instructions
### Running experiments
Experiments can be run from the root folder using the commands below.

Inventory Problem
```
#!/bin/bash

# Create directories
for i in {0..8}; do
  mkdir -p "results/inventory_results/results$i"
done

# Run experiments and plot results
etas=(0.01 0.03 0.05 0.08 0.10 0.15 0.20 0.30)

for i in "${!etas[@]}"; do
    python inventory/inventory_4_LRO_RO.py --foldername "/results/inventory_results/results$i/" --eta "${etas[$i]}"
    python inventory/inventory_8_LRO_RO.py --foldername "/results/inventory_results/results$i/" --eta "${etas[$i]}"
done
python inventory/inventory_4_DRO.py --foldername "/results/inventory_results/results8/" 
python inventory/inventory_8_DRO.py --foldername "/results/inventory_results/results8/"
python inventory/plot_4.py --foldername /results/inventory_results/
python inventory/plot_8.py --foldername /results/inventory_results/
```

Portfolio Optimization
```
#!/bin/bash

# Create directories
for i in {0..16}; do
  mkdir -p "results/portfolio_results/results$i"
done

# Run experiments and plot results
etas=(0.01 0.03 0.05 0.08 0.10 0.15 0.20 0.30)

for i in "${!etas[@]}"; do
    python portfolio/portfolio_5_LRO_RO.py --foldername "/results/portfolio_results/results$i/" --eta "${etas[$i]}"
    python portfolio/portfolio_10_LRO_RO.py --foldername "/results/portfolio_results/results$i/" --eta "${etas[$i]}"
    python portfolio/portfolio_5_MRO.py --foldername "/results/portfolio_results/results$((i+8))/" --eta "${etas[$i]}"
    python portfolio/portfolio_10_MRO.py --foldername "/results/portfolio_results/results$((i+8))/" --eta "${etas[$i]}"
done
python portfolio/portfolio_5_DRO.py --foldername /results/portfolio_results/results16/
python portfolio/portfolio_10_DRO.py --foldername /results/portfolio_results/results16/ 
python portfolio/plot_5.py --foldername /results/portfolio_results/
python portfolio/plot_10.py --foldername /results/portfolio_results/
```
