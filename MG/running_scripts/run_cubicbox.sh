#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4                  # 4 MPI ranks
#SBATCH -c 20                 # 28 threads per rank -> 4*28 = 112 cores total
#SBATCH --mem=20GB               # take full node memory
#SBATCH -t 40:00:00
#SBATCH -p itc_cluster,sapphire
#SBATCH -J test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ---- prep logs dir ----
mkdir -p logs

# ---- user env ----
source ~/.bashrc
conda activate cosmodesi

# ---- modules (match your previous stack) ----
module load cmake/3.25.2-fasrc01
module load gmp/6.2.1-fasrc01
module load mpfr/4.1.0-fasrc01
module load mpc/1.2.1-fasrc01
module load gcc/10.2.0-fasrc01
module load openmpi/4.1.1-fasrc01

# ---- threading (BLAS/OpenMP) ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export PYTHONUNBUFFERED=1

cd /n/home12/cgarciaquintero/DESI/MG_validation/synthetic_noiseless/desilike_runs

srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" python -u run_desilike_validations.py \
  --mock-type cubic+weiliu \
  --tracers ABACUS_MC1_LRG ABACUS_MC1_QSO BGS LRG1 ELG1 ELG2 \
  --ells 0,2,4 \
  --model HDKI \
  --mg-variant binning \
  --redshift-bins --scale-bins \
  --k_S 1.0 --k_TGR 0.0001 --k_tw 0.001 \
  --z_div 1.0 --z_TGR 3.0 --z_tw 0.05 \
  --k_c 0.01 \
  --beyond_eds \
  --prior_basis APscaling \
  --freedom min \
  --kmax-cut 0.20 \
  --kr_max 0.20 \
  --kr_b0_max 0.12 \
  --kr_b2_max 0.08 \
  --nchains "${SLURM_NTASKS}" \
  --chain_name /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains/test.npy \
  --run_chains


###   --mg-variant mu_OmDE \