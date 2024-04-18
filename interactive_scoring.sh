#!/bin/bash
source set_interactive_vars.sh
srun -N 1 --ntasks-per-node=4 --gpus-per-node=4 -u -n 4 python -m earth2mip.lagged_ensembles \
     --model swin_73var_geo_depth12 \
     --output /pscratch/sd/j/jwillard/FCN_exp/earth2mip_swin_validation/swin_73var_geo_depth12/ \
     --start-time 2018-01-02 \
     --end-time 2018-05-29T12 \
     --leads 23 \
     --lags 4

