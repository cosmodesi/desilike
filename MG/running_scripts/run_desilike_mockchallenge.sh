 #!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH --constraint=gpu
#SBATCH -J MG_holi
#SBATCH -o logs/gpu_%x_%j.out
#SBATCH -e logs/gpu_%x_%j.err
#SBATCH -q regular
#SBATCH -A desi_g

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
source activate MGdesi
export PYTHONNOUSERSITE=1
export PYTHONPATH=/global/homes/j/jiaxi/codes_mine/desilike:/global/homes/j/jiaxi/codes_mine/isitgr_private:/global/homes/j/jiaxi/codes_mine/FolpsD:/global/homes/j/jiaxi/codes_mine/cosmoprimo:/global/homes/j/jiaxi/codes_mine/desi-clustering:/global/homes/j/jiaxi/codes_mine/fkptjax_muMG/src:$PYTHONPATH
#
cd /global/homes/j/jiaxi/codes_mine/desilike/MG/running_scripts || exit 1

task=$1
calculator="srun -N 1 -n 1 -C gpu --gpus-per-task=1 --gpu-bind=single:1 -t 04:00:00 --qos shared_interactive --account desi_g"

if [ "$task" = "emu" ]; then
    args="--create-emu"
elif [ "$task" = "run-emu" ]; then
    args="--use-emu --run_chains"
elif [ "$task" = "run" ]; then
    args="--run_chains"
else
    echo "usage: $0 {emu|run-emu|run}"
    exit 1
fi

mg_variant=mu_OmDE
emu_dir=/global/homes/j/jiaxi/codes_mine/desilike/MG/emulators/${mg_variant}_holi
mkdir -p "${emu_dir}"
chains_dir=$SCRATCH/DR2_MG/chains/${mg_variant}_holi_${task}
mkdir -p "${chains_dir}"

prior_bases=(standard physical_velocileptors APscaling)
for prior_basis in "${prior_bases[@]}"; do
    ${calculator} python run_desilike_mockchallenge.py \
        ${args} \
        --mock-type holi_cutsky \
        --tracers LRG1 LRG2 LRG3 ELG1 ELG2 QSO \
        --emu-dir "${emu_dir}" \
        --chain_name "${chains_dir}/fkptjax_holi_${prior_basis}.npy" \
        --ells 0,2 \
        --freedom max \
        --fid-model LCDM \
        --mg-variant ${mg_variant} \
        --prior_basis ${prior_basis} \
        --beyond_eds \
        --restart
done
