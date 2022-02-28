#!/bin/bash
##---------------SLURM Parameters - NLHPC ----------------
#SBATCH -J test_programa4_free
#SBATCH -p general
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8000
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH -o test_programa4_free%j.out
#SBATCH -e test_programa4_free%j.err

#-----------------Toolchain---------------------------
# ----------------Modules----------------------------
ml Python/3.7.3
ml gurobi/9.1.1

# ----------------Command--------------------------
# source venv/bin/activate
cd ../../../
python3 programa.py 3 0 0