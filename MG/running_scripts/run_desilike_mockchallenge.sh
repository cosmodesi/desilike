#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

calculator="srun -N 1 -n 4 -C gpu -t 04:00:00 --qos shared_interactive --account desi_g select_gpu_device"
# MG null test
emu_dir=...
chains_dir=$SCRATCH/DR2_MG/chains
if [ ! -e "${chains_dir}" ]; then
    mkdir -p "${chains_dir}"
fi
args="--emu-dir ${emu_dir} --chains-dir ${chains_dir} --ells 0,2 --mg-variant mu_OmDE --resume"
${calculator} python run_desilike_mockchallenge.py ${args}
