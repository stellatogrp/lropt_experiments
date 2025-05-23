#!/bin/bash
#SBATCH --job-name=invtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=20G
#SBATCH --time=23:40:00
#SBATCH -o /scratch/gpfs/iywang/lropt_revision/output/inventory/inv_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0        # job array

module purge
module load anaconda3/2023.9
conda activate lropt_rev

python lropt_experiments/inventory_parallel/inv_bi1.py --config-name=inv_bi.yaml
# python lropt_experiments/inventory_parallel/inv_dro.py --config-name=inv_dro.yaml
#--config-name=port.yaml seed=10 eta=10 

# python portfolio/plot_avg_10.py
