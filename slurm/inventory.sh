#!/bin/bash
#SBATCH --job-name=invtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/learn_robust/lropt_results/inventory/inv_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

module purge
module load anaconda3/2023.9
conda activate lropt

# python inventory/inventory_8_MRO.py --foldername /scratch/gpfs/iywang/learn_robust/lropt_results/inventory/old56/results15/ --eta 0.30

python inventory/plot_avg_redo.py --foldername /scratch/gpfs/iywang/learn_robust/lropt_results/inventory/old46/

# python portfolio/MIP/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/new/m30_K1000_r10/

# 0 m=4 r=0 lr=0.000005 iters=1000 y=0.3 seed 27, mu_mult = 1.01
# 1 m=4 y=0.3 seed 5
# 2 m=4 y=0.3 seed 15
# 3 seed 20
# 4 seed 10
# 5 seed 50
# 6 seed 50, 20 update

# 0-lam-1.5,mu-1,s50, 0.8*init, init+0.5*rand + 0.01I
# 1-s5
# 2-s15
# 3-s30
# 4-s40

# 0 0.1*init scene + 3, seed = 100, 200, 
# 2 0.1*init seed = 300, 400, 500
# 5 rand(30,40) seed 0

# 0 0.1*init scene + 3, rand(20,40), stepsize = 0.0001, seed = 100,200,300,400
# seed = 0, 20, 
# seed = 0,20,stepsize = 0.00001
# seed = 0,20,stepsize = 0.000005


# 0 0.01*init scene + 3, 27, stepsize = 0.00001, seed = 0,20,30,40,50,60

# 0 0.001*init scene + 3, 27, stepsize = 0.001, seed = 0,10,20
# stepsize = 0.0001 seed = 0,10,20


# 0 m=6 0.3*init scene + 3, 27, stepsize = 0.00001, seed = 0,10,20,30
# m=2, seed= 0, 10


# 0 m=2, eta= 0.1, seed = 0
# 1 m=4, eta= 0.1, seed = 0, 10
# 3 m=2 eta=0.2, seed = 0, init lam mu = 0.1
# 4 m=3, m=4, m=5
# 7 m=4, seed=0.0001, eta=0.15
# 8 m=4 eta=0.3, m=2
# 10 eta=0.4, m=4


# 0 m=4, eta=0.05, seed = 5, 15
# 2 m=4, eta=0.2, seed=5,15
# 4 m=2, eta=0.05, seed=5, m=8
# 6 m=4, eta=0.01, seed=5,15
# 8 m=4, eta=0.4, seed=5,15
# 10 m=4, eta=0.2, seed=5,15


# 0 m=4, eta=0.01, seed=15, eta=0.05, 0.1,0.2,0.3,0.4 lam,mu=1,1.5

# 0 m=4, eta=0.01, seed=25, eta=0.05, 0.1,0.2,0.3,0.4

# (old13) lr=0.001, init, eta=0.01, 0.03, .05,.1,.15,.2, lam,mu=0.5,0.5

# lr =0.005, init, seed=25, 0.5I+0.5rand

# old16 newparam, lr=0.0001, lam,mu=0.5,0.5. eta=0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, seed=27

# old17 lr=0.0005, seed=25
# old19 single constraint stepsize 100, rate 0.2
# old20 eps=5, lr=0.0005 
#old21 eps=1 lr=0.0001 lam,mu=1  stepsize 50, rate 0.5
#old22 eps=1 lr=0.0001, lam,mu=2 stepsize50 rate 0.5
#old23 lam,mu=3

#old24 m=8 lam,mu=2
#old25 m=6 lam,mu=2 0.01 .03 .05 .08 0.9 .10 .15 .2 .3

#old26 m=4 lam,mu=2, initA=100*init, initA=20*init both, 100 both lam 0.9
#old27 m=4 initA=10*init, lam 0.99, //initA=1*init, lam mu both update mu_mult 1.015

#old28 m=4 initA=10*init, lam 1.1, mu every 30, mu_mult=1.02

#old29 m=8, initA=10*init, lam every 20, mu every 30, mu_mult=1.02

#old30 m=4 initA=10*init, all y's, 1000 test, smaller batches
#old31 m=4 initA=10*init, parallel false, rep y/ all y, batch=40
#old32 m=4 10*init, parrallel 6 true all y, rep y
#old33 m=4 10*init, no parallel, all y's, rep y, 8000 test
#old34 m=8 10*init, no parallel, rep y's, all y, 5000 test, batch 20

#old35 m=4 10*init, no parallel, 10r, ally's, 5000 test, batch 30, repy 20000/5
#old36 m=8 7r, m=10, n=20 k=0.1

#old37 m10n20 7r init, no parallel, with parallel
#old38 m4 10r 5*init no parallel 5repy k=0.5, init

#old39 m4 20r 5*init no parallel 10repy k=0.5, m8 15r (CURRENT)

#old40 m8 nopar r20 5*init k=0.,/ init k=0. (CURRENT)
#old41 m4 nopar r5 init I mro K=200 k=0./, m8 r5 init I / K=800 m8 m4

#old42 m8 r20 init k=0 traineps 200iters lr0.01 no par, m4
#old43 m4 r20 5*init k=0. lr0.0001, m8
#old44 m8 init lr=0.0001, m4 + 0.03 with train
#old45 m8 init lr=0.001, m4 

#old46 m8 init // m4 N=100 //mro (CURRENT)

#old47 mro m4 0.5init // m8  lam,mu=2 , lr=0.0001
#old48 mro m4//m8 0.5init, lam200,mu=200, lr=0.0001
#old49 mro m4 0.3init, lam100, mu200, lr=0.0005 kappa=0.1 // lam,mu100 lr=0.001 k0.05 0.2init
#old50 mro m8 0.3init, lam100mu200 lr=0.0005 k0.1 // m4 1.5init lammu20, lr=0.0001 k0.01
#old51 m8 1.5init lammu20, lr=0.0001 k0.01 // m4 0.5*init, lammu20,lr=0.001 k=0.01
#old52 m8 0.5init, lammu20, lr=0.001, k=0.01 //0.4init, lammu30, lr=0.001, k=0.1
#old53 m4 0.4init, lammu30,lr=0.001,k=0.1 // m4 0.4init, k=0.05 r+10
#old54 m8 0.4init k=0.05 r+10 (USABLE)// m8 0.4init, k=0.05, r+10, lammu30, lr=0.01
#old55 m4 0.4init, k=0.05, r+10, lammu30, lr=0.01 // 
#old56 m4 mvinit, lr=0.001 k=0.1 // m8 mvinit, k=0.1 lr=0.001 lammu30