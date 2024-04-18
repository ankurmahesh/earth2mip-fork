#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH -C 'gpu&hbm80g'
#SBATCH --account=m4416
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -J swin_73var_depth24_e2048_mlp2_chweight_invar_00_0
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jwillard@lbl.gov
#SBATCH -o scoring%x-%j.out

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0


set -x
cd /global/u2/j/jwillard/earth2mip-fork
source set_interactive_vars.sh
srun -N 4 --ntasks-per-node=4 --gpus-per-node=4 -u -n 1 python -m earth2mip.lagged_ensembles --model swin_73var_depth24_e2048_mlp2_chweight_invar_00 --output /pscratch/sd/j/jwillard/FCN_exp/earth2mip_swin_validation/lagged_ensembles/swin_73var_depth24_e2048_mlp2_chweight_invar_00/ --start-time 2018-01-02 --end-time 2018-05-29T12 --leads 29 --lags 4
