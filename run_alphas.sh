#!/bin/bash
#SBATCH --job-name=run_alphas
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --partition=cpu_long

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

# To clean and load modules defined at the compile and link phases
module purge
module restore code_env

source /gpfs/users/serreg/.bashrc
conda activate dev

# echo of commands
set -x

cd ${SLURM_SUBMIT_DIR}

# execution
python run_alphas.py