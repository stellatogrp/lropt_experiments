#!/bin/bash
#SBATCH --job-name=porttest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --time=23:40:00
#SBATCH -o /scratch/gpfs/iywang/lropt_revision/output/portfolio/port_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com
#SBATCH --array=0-19       # job array

module purge
module load anaconda3/2023.9
conda activate lropt_rev

python lropt_experiments/LCX/LCX_sep4.py --config-name=lcx4.yaml
# python lropt_experiments/port_parallel/port_dro.py --config-name=port_dro.yaml
# python lropt_experiments/port_parallel/port_delage.py --config-name=port_delage.yaml

#--config-name=port.yaml seed=10 eta=10 

# python portfolio/plot_avg_10.py
