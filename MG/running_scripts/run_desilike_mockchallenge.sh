#!/bin/bash
#SBATCH -N 2
#SBATCH -n 8                  # 4 GPU ranks
#SBATCH -c 32                 
#SBATCH -t 19:00:00
#SBATCH --constraint=cpu
#SBATCH -J MG_test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -q regular
#SBATCH -A desi

# create a new environment, install desilike and pyfkpt locally with pip
source activate MGdesi
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
export PYTHONPATH=/global/homes/j/jiaxi/codes_mine/desilike:$PYTHONPATH

task=$1
Nnode=4
#calculator="srun -N 1 -n ${Nnode} --gpus-per-task=1 -C gpu -t 04:00:00 --qos interactive --account desi_g select_gpu_device"
#calculator="srun -N 1 -n 8 -c 16 -C cpu -t 04:00:00 --qos shared_interactive --account desi "
#calculator="srun -N 2 -n 8 -c 32 "
if [ "$task" = "emu" ]; then
    args="--create-emu --emu-order 3 "  
elif [ "$task" = "run-emu" ]; then
    args="--use-emu "
elif [ "$task" = "run" ]; then
    args=""
else
    echo "the input should be: emu, run-emu, run"
    exit
fi
# MG null test
mg_variant=mu_OmDE
emu_dir=/global/homes/j/jiaxi/codes_mine/desilike/MG/emulators/${mg_variant}
if [ ! -e "${emu_dir}" ]; then
    mkdir -p "${emu_dir}"
fi
chains_dir=$SCRATCH/DR2_MG/chains/${mg_variant}
if [ ! -e "${chains_dir}" ]; then
    mkdir -p "${chains_dir}"
fi

prior_bases=(standard physical_velocileptors APscaling)
#for prior_basis in "${prior_bases[@]}"; do
for ((j=0; j<=0; j++)); do
    ${calculator} python run_desilike_mockchallenge.py \
    ${args} \
    --emu-dir ${emu_dir} \
    --chains-dir ${chains_dir} \
    --ells 0,2 \
    --freedom min \
    --fid-model LCDM \
    --mg-variant ${mg_variant} \
    --prior-basis ${prior_bases[${j}]} \
    --beyond-eds \
    --resume \
    --chain-prefix "fkptjax_mcmc_${prior_bases[${j}]}" \
    #> ./fkptjax_${prior_bases[${j}]}.log
done
