#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4                  # 4 MPI ranks
#SBATCH -c 64
#SBATCH -t 40:00:00
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J m1_emu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ---- prep logs dir ----
mkdir -p logs

# ---- user env ----
source ~/.bashrc
cosmodesi_ide

# ---- threading (BLAS/OpenMP) ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export PYTHONWARNINGS=once
# export PYTHONUNBUFFERED=1

cd /global/homes/n/nishavk/desilike/IDE/running_scripts/

python -u ide_running_script.py \
  --mock-type cubic+weiliu \
  --tracers ABACUS_MC1_LRG ABACUS_MC1_QSO BGS LRG1 ELG1 ELG2 \
  --ells 0,2,4 \
  --model IDE \
  --ide-variant IDEModel1 \
  --beyond_eds \
  --prior_basis APscaling \
  --freedom min \
  --kmax-cut 0.20 \
  --kr_max 0.20 \
  --kr_b0_max 0.12 \
  --kr_b2_max 0.08 \
  --create-emu \
  --emu-order-lcdm 2 \
  --emu-order-ide 4 \
  --emu-dir /global/homes/n/nishavk/desilike/IDE/emulators/
