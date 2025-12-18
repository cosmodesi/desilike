#!/bin/bash

# create a new environment, install desilike and pyfkpt locally with pip
source activate MGtest
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=/global/homes/j/jiaxi/codes_mine/desilike:$PYTHONPATH

task=$1
if [ "$task" = "emu" ]; then
    Nnode=1
    args="--create-emu "
elif [ "$task" = "run-emu" ]; then
    Nnode=4
    args="--use-emu "
elif [ "$task" = "run" ]; then
    Nnode=4
    args=""
fi
calculator="srun -N 1 -n ${Nnode} -C gpu -t 04:00:00 --qos shared_interactive --account desi_g select_gpu_device"
# MG null test
mg_variant=mu_OmDE
emu_dir=/global/homes/j/jiaxi/codes_mine/desilike/MG/emulators/${mg_variant}
chains_dir=$SCRATCH/DR2_MG/chains/${mg_variant}
if [ ! -e "${chains_dir}" ]; then
    mkdir -p "${chains_dir}"
fi

${calculator} python run_desilike_mockchallenge.py \
    ${args} \
    --emu-dir ${emu_dir} \
    --chains-dir ${chains_dir} \
    --ells 0,2 \
    --freedom min \
    --fid-model LCDM \
    --MG-model HDKI \
    --mg-variant ${mg_variant} \
    --priors-basis physical_velocileptors \
    --beyond-eds \
    --resume \
    --chain-prefix "isitgr_fkpt_noextf_new" \
#${args}
