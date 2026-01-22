#!/bin/bash
#SBATCH --job-name=MGtest
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mem=80G
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH -c 2
#
# create a new environment, install desilike and pyfkpt locally with pip
#conda init
#conda activate MGdesi
export PYTHONPATH=/home/jiaxiyu/codes/desilike:$PYTHONPATH

task=$1
Ntask=4
calculator="srun -N 1 -n ${Ntask} -c 2 --gres=gpu:${Ntask} --gpu-bind=single:1 -p main -t 24:00:00 --mem=80G"

if [ "$task" = "emu" ]; then
    args="--create-emu --emu-order 3 "  
    sufix=""
elif [ "$task" = "run-emu" ]; then
    args="--use-emu "
    sufix="_with-emu"
elif [ "$task" = "run" ]; then
    args=""
    sufix="_no-emu"
else
    echo "the input should be: emu, run-emu, run"
    exit
fi
# MG null test
mg_variant=mu_OmDE
emu_dir=$SCRATCH/MGtest/emulators/${mg_variant}
if [ ! -e "${emu_dir}" ]; then
    mkdir -p "${emu_dir}"
fi
chains_dir=$SCRATCH/MGtest/chains/${mg_variant}_${task}
if [ ! -e "${chains_dir}" ]; then
    mkdir -p "${chains_dir}"
fi

prior_bases=(standard physical_velocileptors APscaling)
#for prior_basis in "${prior_bases[@]}"; do
for ((j=0; j<=2; j++)); do
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
