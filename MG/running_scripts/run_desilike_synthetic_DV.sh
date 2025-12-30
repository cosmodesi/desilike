#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4                  # 4 MPI ranks
#SBATCH -c 28                 # 28 threads per rank -> 4*28 = 112 cores total
#SBATCH --mem=50GB               # take full node memory
#SBATCH -t 00:30:00
##48:00:00
#SBATCH -p test
##itc_cluster,sapphire
#SBATCH -J mcmc_LCDM_emu_phys_mu0_beds_min_standard
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

cd /n/home12/cgarciaquintero/DESI/MG_validation/synthetic_noiseless/desilike_runs/

srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
  python -u run_desilike_synthetic_data.py \
    --data-tag LCDM \
    --mode emcee \
    --ells 0,2 \
    --freedom min \
    --fid-model LCDM \
    --mg-variant mu_OmDE \
    --beyond-eds \
    --prior-basis standard \
    --chain-prefix "fkptjax" \
    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains \
    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

#python -u run_desilike.py --use-emu --ells 0,2 --freedom max --fid-model LCDM --MG-model HDKI --mg-variant mu_OmDE --prior-basis APscaling --chain-prefix "test" --mode mcmc --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains  --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

# RUN MINIMIZATION

#srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
#  python -u run_desilike.py \
#    --use-emu \
#    --ells 0,2 \
#    --freedom min \
#    --model LCDM \
#    --redshift-bins \
#    --mode map \
#    --chain-prefix "chain_fs_direct_folps_isitgr" \
#    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains  \
#    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

# BUILD EMULATOR

#srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
#  python -u run_desilike.py \
#    --create-emu \
#    --ells 0,2 \
#    --freedom max \
#    --model LCDM \
#    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains  \
#    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

#srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
#  python -u run_desilike.py \
#    --create-emu \
#    --ells 0,2,4 \
#    --freedom min \
#    --model LCDM \
#    --redshift-bins \
#    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains  \
#    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

#srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
#  python -u run_desilike.py \
#    --create-emu \
#    --ells 0,2,4 \
#    --freedom min \
#    --model LCDM \
#    --redshift-bins --scale-bins --kc 0.01 \
#    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains  \
#    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators

#srun --mpi=pmix -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" \
#  python -u run_desilike.py \
#    --create-emu \
#    --ells 0,2 \
#    --freedom min \
#    --model LCDM \
#    --redshift-bins --scale-bins --kc 0.1 \
#    --chains-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/chains \
#    --emu-dir /n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators