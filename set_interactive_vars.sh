export MASTER_ADDR=$(hostname)
export MODEL_REGISTRY=/pscratch/sd/a/amahesh/earth2mip_model_registry/
export MASTER_PORT=29500
conda activate /global/common/software/m4416/fcn_mip-env/
export ERA5_HDF5=/pscratch/sd/p/pharring/73var-6hourly/staging/



#srun --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-node=4 --gpu-bind=map_gpu:0,1,2,3 -n 4 python3 -u -m fcn_mip.inference_ensemble interactiveq_config.json
