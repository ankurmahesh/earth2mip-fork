export MASTER_ADDR=$(hostname)
export MODEL_REGISTRY=/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/
export MASTER_PORT=29500
module load conda
conda activate /global/cfs/cdirs/m4416/jared/fcn_mip-env_e2mip_update/
export ERA5_HDF5=/pscratch/sd/p/pharring/73var-6hourly/staging/
export CUDA_VISIBLE_DEVICES=3,2,1,0

