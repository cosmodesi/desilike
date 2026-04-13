#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -q regular
#SBATCH -A desi_g
#SBATCH -J horndeski_smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# The DESI cosmodesi setup already puts the stack's base conda environment on PATH.
# Only activate another environment if explicitly requested, and initialize conda
# first so this works in non-interactive SLURM shells.
if [ -n "${CONDA_ENV:-}" ]; then
  CONDA_BASE="${CONDA_BASE:-/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/conda}"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export FOLPS_BACKEND="${FOLPS_BACKEND:-jax}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

BASE_DIR="${BASE_DIR:-/global/u2/j/jiamingp/HEFTCAMB/HEFTCAMB_DESI}"
export BASE_DIR
export PYTHONPATH="${BASE_DIR}/EFTCAMB:${BASE_DIR}/desilike_fkptdev:${BASE_DIR}/fkptjax_muMG/src:${BASE_DIR}/FolpsD:${PYTHONPATH:-}"

cd "${BASE_DIR}" || exit 1

CHAINS_DIR="${CHAINS_DIR:-${BASE_DIR}/Test/chains_horndeski_direct}"
export CHAINS_DIR
mkdir -p "${CHAINS_DIR}"

echo "Running on host: $(hostname)"
echo "BASE_DIR=${BASE_DIR}"
echo "CHAINS_DIR=${CHAINS_DIR}"
NTASKS="${NTASKS:-${SLURM_NTASKS:-1}}"
echo "NTASKS=${NTASKS}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-1}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}"

python - <<'PY'
import os
import sys
import jax
print("JAX devices:", jax.devices())

base = os.environ["BASE_DIR"]
for path in [f"{base}/EFTCAMB", f"{base}/desilike_fkptdev", f"{base}/fkptjax_muMG/src", f"{base}/FolpsD"]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
for name in list(sys.modules):
    if name == "camb" or name.startswith("camb.") or name == "desilike" or name.startswith("desilike.") or name == "fkptjax" or name.startswith("fkptjax."):
        del sys.modules[name]
import camb
import desilike
import fkptjax
print("python:", sys.executable)
print("camb:", camb.__file__)
print("desilike:", desilike.__file__)
print("fkptjax:", fkptjax.__file__)
PY

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: environment/import check passed; not launching srun."
  exit 0
fi

srun -N 1 -n "${NTASKS}" -c "${SLURM_CPUS_PER_TASK:-16}" \
  --gpus-per-task=1 --gpu-bind=single:1 \
  python -u desilike_fkptdev/MG/running_scripts/run_desilike_horndeski_synthetic_DV.py \
    --base-dir "${BASE_DIR}" \
    --mode "${MODE:-mcmc}" \
    --direct-eftcamb \
    --vary-cosmology \
    ${VARY_ALPHAM:+--vary-alphaM} \
    --vary-alphaB \
    --synthetic-from-theory \
    --nchains "${NCHAINS:-${NTASKS}}" \
    --max-iter "${MAX_ITER:-20}" \
    --check-every "${CHECK_EVERY:-10}" \
    --max-eigen-gr "${MAX_EIGEN_GR:-0.2}" \
    --stable-over "${STABLE_OVER:-1}" \
    --tracer "${TRACER:-BGS}" \
    --ells "${ELLS:-0,2}" \
    --kmin-cut "${KMIN_CUT:-0.02}" \
    --kmax-cut "${KMAX_CUT:-0.20}" \
    --alphaM0 "${ALPHAM0:-0.1}" \
    --alphaB0 "${ALPHAB0:-0.0}" \
    --alphaK0 "${ALPHAK0:-0.5}" \
    --vary-nuisance "${VARY_NUISANCE:-all}" \
    --chains-dir "${CHAINS_DIR}" \
    ${EXTRA_ARGS:-}
