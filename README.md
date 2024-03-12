# lropt_experiments
This repository is by
[Irina Wang](https://sites.google.com/view/irina-wang),
Amit Soloman,
[Bart Van Parys](https://mitsloan.mit.edu/faculty/directory/bart-p-g-van-parys),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Learning Decision-Focused Uncertainty in Robust Optimization]([http://arxiv.org/abs/](https://arxiv.org/abs/2305.19225))."

If you find this repository helpful in your publications,
please consider citing our paper.

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
pip install lropt
```

## Instructions
### Running experiments
Experiments can from the root folder using the commands below.

Inventory Problem (for m=8). For m=4, replace 8 with 4 in the following file names. 
```
python inventory/make_dir.py
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results0/ --eta 0.01
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results1/ --eta 0.03
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results2/ --eta 0.05
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results3/ --eta 0.08
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results4/ --eta 0.10
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results5/ --eta 0.15
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results6/ --eta 0.20
python inventory/inventory_8_LRO_RO.py --foldername /results/inventory_results/results7/ --eta 0.30
python inventory/inventory_8_DRO.py --foldername /results/inventory_results/results8/ --eta 0.30
python inventory/plot_8.py --foldername /results/inventory_results/
```
Portfolio Optimization (for m=10). For m=5, replace 10 with 5 in the following file names. 
```
python portfolio/make_dir.py
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results0/ --eta 0.01
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results1/ --eta 0.03
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results2/ --eta 0.05
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results3/ --eta 0.08
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results4/ --eta 0.10
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results5/ --eta 0.15
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results6/ --eta 0.20
python portfolio/portfolio_10_LRO_RO.py --foldername /results/portfolio_results/results7/ --eta 0.30
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results8/ --eta 0.01
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results9/ --eta 0.03
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results10/ --eta 0.05
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results11/ --eta 0.08
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results12/ --eta 0.10
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results13/ --eta 0.15
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results14/ --eta 0.20
python portfolio/portfolio_10_MRO.py --foldername /results/portfolio_results/results15/ --eta 0.30
python portfolio/portfolio_10_DRO.py --foldername /results/portfolio_results/results16/ --eta 0.30
python portfolio/plot_10.py --foldername /results/portfolio_results/
```
