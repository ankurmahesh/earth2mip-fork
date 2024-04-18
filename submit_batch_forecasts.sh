#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH -C 'gpu&hbm80g'
#SBATCH --account=m4416
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J swin_73var_geo_depth12_e1536_forecasts2018
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jwillard@lbl.gov
#SBATCH -o scoring%x-%j.out

export MODEL_REGISTRY=/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/
export MASTER_PORT=29500
module load conda
conda activate /global/cfs/cdirs/m4416/jared/fcn_mip-env_e2mip_update/
export ERA5_HDF5=/pscratch/sd/p/pharring/73var-6hourly/staging/

export CUDA_VISIBLE_DEVICES=3,2,1,0
srun -N 1 -l --ntasks-per-node=4 --gpus-per-node=4 -u -n 4 python -u wb2_outputs_e1536_8step_geo.py