import json
import os
import datetime
import sys
import urllib
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Any, Iterator, List, Callable
import numpy as np
import xarray as xr
# Get the current working directory
current_dir = Path(os.getcwd())

# Assuming earth2mip is a sibling directory to the current directory
# Get the parent directory of the current working directory
parent_dir = current_dir.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from earth2mip.networks import _load_package
from earth2mip import schema, model_registry, ModelRegistry, time_loop
from earth2mip.initial_conditions import hdf5
from earth2mip.time_collection import run_over_initial_times
from earth2mip.schema import EnsembleRun



def get_model(
    model: str,
    registry: ModelRegistry = None,
    device="cpu",
    metadata: Optional[schema.Model] = None,
) -> time_loop.TimeLoop:
    """
    Function to construct an inference model and load the appropriate
    checkpoints from the model registry

    Parameters
    ----------
    model : The model name to open in the ``registry``. If a url is passed (e.g.
        s3://bucket/model), then this location will be opened directly.
        Supported urls protocols include s3:// for PBSS access, and file:// for
        local files.
    registry: A model registry object. Defaults to the global model registry
    metadata: If provided, this model metadata will be used to load the model.
        By default this will be loaded from the file ``metadata.json`` in the
        model package.
    device: the device to load on, by default the 'cpu'


    Returns
    -------
    Inference model


    """
    url = urllib.parse.urlparse(model)

    if url.scheme == "e2mip":
        package = registry.get_model(model)
        return _load_package_builtin(package, device, name=url.netloc)
    elif url.scheme == "":
        package = registry.get_model(model)
        return _load_package(package, metadata, device)
    else:
        package = model_registry.Package(root=model, seperator="/")
        return _load_package(package, metadata, device)

#load data
res_dir = '/pscratch/sd/j/jwillard/FCN_exp/earth2mip_swin_validation/scores_ams/' 
plot_dir = '/pscratch/sd/j/jwillard/FCN_exp/earth2mip_swin_validation/plots/' 

from earth2mip.inference_medium_range import save_scores, time_average_metrics, score_deterministic

with open("./config_swin_depth12_chweight_inv_8step.json") as f:
    config_tmp = json.load(f)
    
h5_folder = "/pscratch/sd/p/pharring/73var-6hourly/staging/"
registry = model_registry.ModelRegistry('/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/')
if __name__ == '__main__':
    gpu_id = int(os.environ.get('SLURM_LOCALID', '0'))
    device = f'cuda:{gpu_id}'
    model = get_model(config_tmp['weather_model'], registry, device=device)

    time = datetime.datetime(2018, 1, 1, 0)
    # initial_times = [time + datetime.timedelta(hours=12 * i) for i in range(730)]
    initial_times = [time + datetime.timedelta(hours=12 * i) for i in range(10)]

    datasource = hdf5.DataSource.from_path(
        root=h5_folder, channel_names=model.channel_names
    )

    time_mean = np.load('/pscratch/sd/p/pharring/73var-6hourly/staging/stats/time_means.npy')
    config_path = '../config_swin_depth12_chweight_inv_8step.json'
    output_path = '../pscratch/sd/j/jwillard/FCN_exp/wb2_forecasts/DEBUGswin_73var_geo_depth12_chweight_invar_8step/'
    with open(config_path) as f:
        config_geo_chw_8step = json.load(f)
    config = EnsembleRun.parse_obj(config_geo_chw_8step)
    n_shards = 4

    # #just this line for single node
    shard = int(os.environ.get('SLURM_LOCALID', '0'))

    #multi-node run this
    # node_id = int(os.environ.get('SLURM_NODEID', '0'))
    # tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', '4'))
    # gpu_id = int(os.environ.get('SLURM_LOCALID', '0'))
    # shard = node_id * tasks_per_node + gpu_id

    print("executing on device: ", device, ", w/ shard ", shard)

    run_over_initial_times(time_loop=model, data_source=datasource, 
                        initial_times=initial_times, 
                        config=config, output_path=output_path, 
                        shard=shard,n_shards=n_shards, score=False)



