#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=15G
#SBATCH --time=23:40:00
#SBATCH -o /scratch/gpfs/iywang/lropt_revision/output/portfolio/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0     # job array

module purge
module load anaconda3/2023.9
conda activate lropt_rev

python lropt_experiments/port_parallel/port_knn_bi.py --config-name=port_bi1.yaml
# python lropt_experiments/port_parallel/port_dro.py --config-name=port_dro.yaml
# python lropt_experiments/port_parallel/port_delage2.py --config-name=port_delage4.yaml

#--config-name=port.yaml seed=10 eta=10 

# python portfolio/plot_avg_10.py
