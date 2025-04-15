#!/bin/bash
#SBATCH -N 6
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -t 00:35:00
#SBATCH -A m4416
#SBATCH --job-name=hens_percentile_crps
#SBATCH --output=logs_nobootstrap_hens_crps/t2m_h5_reduce_%j.log
#SBATCH --array=0-91

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE

source deactivate
module load conda
conda deactivate
conda activate /global/common/software/m4416/fcn_mip-env/
srun -u -N 6 --ntasks-per-node=2 -c 64 --cpu-bind=cores --gpus-per-node=2 python -u nobootstrap_hens_crps.py --variable t2m --percentile 95 --slurm_array_id $SLURM_ARRAY_TASK_ID --slurm_array_size $SLURM_ARRAY_TASK_COUNT 

