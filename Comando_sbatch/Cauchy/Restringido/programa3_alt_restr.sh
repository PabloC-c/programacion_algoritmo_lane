#!/bin/bash
##---------------SLURM Parameters - NLHPC ----------------
#SBATCH -J test_programa3_alt_restr
#SBATCH -p general
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8000
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH -o test_programa3_alt_restr%j.out
#SBATCH -e test_programa3_alt_restr%j.err

#-----------------Toolchain---------------------------
# ----------------Modules----------------------------
ml Python/3.7.3
ml gurobi/9.1.1

# ----------------Command--------------------------
# source venv/bin/activate
cd ../../../
python3 programa_alt.py 2 1 1