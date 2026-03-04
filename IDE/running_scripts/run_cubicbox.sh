#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4                  # 4 MPI ranks
#SBATCH -c 64
#SBATCH -t 40:00:00
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J StMaxAll
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ---- prep logs dir ----
mkdir -p logs

# ---- user env ----
source ~/.bashrc
cosmodesi_ide

# ---- threading (BLAS/OpenMP) ----
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export PYTHONWARNINGS=once

cd /global/homes/n/nishavk/desilike/IDE/running_scripts/

srun -n 4 -c 64 --cpu_bind=cores python -u ide_running_script.py \
  --mock-type cubic+weiliu \
  --tracers ABACUS_MC1_LRG ABACUS_MC1_QSO BGS LRG1 ELG1 ELG2 \
  --ells 0,2,4 \
  --model IDE \
  --ide-variant IDEModel1 \
  --beyond_eds \
  --prior_basis standard \
  --freedom max \
  --kmax-cut 0.20 \
  --kr_max 0.20 \
  --kr_b0_max 0.12 \
  --kr_b2_max 0.08 \
  --nchains 4 \
  --chain_name /pscratch/sd/n/nishavk/DR2_FullShape/cubic_weiliu_beyondEdS_standard_maximum_alltracers/test.npy \
  --run_chains \
  --restart \
  --check-every 500