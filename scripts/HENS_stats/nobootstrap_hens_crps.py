"""
This script calculates the number of members of the 7424 member ensemble and the 58 member ensemble that are 
            1. above the 95th percentile threshold

It also calculates 
            4. CRPS
            5. twCRPS
            6. owCRPS
for the 7424 member ensemble and the 58 member ensemble

This script only uses one bootstrap trial per ensemble size:
- The full 7424 member ensemble size (no random sampling with replacement, just the 7424 members)
- The 58 member described in the HENS part 1 paper.  The 58 members are from 1 centered bred vector per 29 model checkpoints

A sample usage on 3 Perlmutter GPU nodes is 

conda activate /global/common/software/m4416/fcn_mip-env/
srun --gpus-per-node=4 -N 3 -u --ntasks-per-node=4 --cpu-bind=cores -c 32 python -u nobootstrap_hens_crps.py --variable t2m --slurm_array_id 0 --slurm_array_size 92 --percentile 95

on 12 perlmutter nodes, the usage is (using shifter):
srun --mpi=pmi2 --gpus-per-node=1 -N 12 -u --ntasks-per-node=1 --cpu-bind=cores -c 128 shifter --image=registry.nersc.gov/dasrepo/pharring/deepspeed-pytorch:24.04 --module gpu python -u nobootstrap_hens_crps.py --variable t2m --slurm_array_id $SLURM_ARRAY_TASK_ID --slurm_array_size 92 --percentile 95
"""

from mpi4py import MPI
import h5py as h5
import xarray as xr
import argparse
import pandas as pd
from datetime import timedelta
import numpy as np
from timeit import default_timer
#from modulus.metrics.general import crps
import torch
import os

@torch.jit.script
def crps(
    pred: torch.Tensor, obs: torch.Tensor, dim: int = 0, method: str = 'sort',
) -> torch.Tensor:
    """Compute the exact CRPS using the CDF method

    This is directly taken from the official NVIDIA Modulus codebase: https://github.com/NVIDIA/physicsnemo/blob/353b3faa54454a2230529189bb05b4d6ed8f9b98/modulus/metrics/general/crps.py

    Uses this formula
    .. math::
        \\int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.

    This method is more memory efficient than the kernel method, and uses O(n
    log n) compute instead of O(n^2), where n is the number of ensemble members.

    Parameters
    ----------
    pred : torch.Tensor
        tensor of ensemble members / predictions
    obs : torch.Tensor
        tensor of observations
    dim : int
        Dimension to perform CRPS reduction over.

    Returns
    -------
        tensor of CRPS scores

    """
    n = pred.shape[dim]
    device = pred.device
    pred, _ = torch.sort(pred, dim=dim)
    ans = torch.zeros_like(obs)

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    # val = ensemble[0] - truth
    val = (
        torch.index_select(
            pred, dim, torch.tensor([0], device=device, dtype=torch.int32)
        ).squeeze(dim)
        - obs
    )
    ans += torch.where(val > 0, val, 0.0)

    for i in range(n - 1):
        x0 = torch.index_select(
            pred, dim, torch.tensor([i], device=device, dtype=torch.int32)
        ).squeeze(dim)
        x1 = torch.index_select(
            pred, dim, torch.tensor([i + 1], device=device, dtype=torch.int32)
        ).squeeze(dim)

        cdf = (i + 1) / n

        # a. case y < x0
        val = (x1 - x0) * (cdf - 1) ** 2
        mask = obs < x0
        ans += torch.where(mask, val, 0.0)

        # b. case x0 <= y <= x1
        val = (obs - x0) * cdf**2 + (x1 - obs) * (cdf - 1) ** 2
        mask = (obs >= x0) & (obs <= x1)
        ans += torch.where(mask, val, 0.0)

        # c. case x1 < t
        mask = obs > x1
        val = (x1 - x0) * cdf**2
        ans += torch.where(mask, val, 0.0)

    # dx [F(x) - H(x-y)]^2 = dx [1 - 0]^2 = dx
    val = obs - torch.index_select(
        pred, dim, torch.tensor([n - 1], device=device, dtype=torch.int32)
    ).squeeze(dim)
    ans += torch.where(val > 0, val, 0.0)
    return ans


@torch.jit.script
def _owcrps(
    pred: torch.Tensor, obs: torch.Tensor, thresh: torch.Tensor, dim: int = 0, 
):
    """Compute the exact owCRPS using the CDF method

    Uses this formula
    .. math::
        \\int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.

    This method is more memory efficient than the kernel method, and uses O(n
    log n) compute instead of O(n^2), where n is the number of ensemble members.

    Parameters
    ----------
    pred : torch.Tensor
        tensor of ensemble members / predictions
    obs : torch.Tensor
        tensor of observations
    dim : int
        Dimension to perform CRPS reduction over.

    Returns
    -------
        tensor of CRPS scores

    """
    n = pred.shape[dim]
    device = pred.device
    pred, _ = torch.sort(pred, dim=dim)
    ans = torch.zeros_like(obs)
    
    obs_above_thresh = obs > thresh
    num_ens_above_thresh = (pred > torch.unsqueeze(thresh, dim)).sum(dim=dim)

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    # val = ensemble[0] - truth
    #first_member = torch.index_select(
    #       pred, dim, torch.tensor([0], device=device, dtype=torch.int32)
    #   ).squeeze(dim)

    #Select the first member of pred that is greater than thresh
    # Assuming pred is your input tensor with dimensions (721, 58, 1440)
    # and thresh is your threshold tensor with dimensions (721, 1440)

    # Expand thresh to match the dimensions of pred
    thresh_expanded = thresh.unsqueeze(1).expand_as(pred)

    # Create a mask where pred is greater than thresh
    mask = (pred > thresh_expanded).int()

    # Find the index of the first True value along the ensemble dimension
    first_greater_indices = mask.argmax(dim=1)

    # Gather the corresponding values from pred
    first_member = pred.gather(1, first_greater_indices.unsqueeze(1)).squeeze(1)

    # first_greater_values now contains the first value greater than thresh for each (lat, lon) pair

    val = first_member - obs
    ans += torch.where(torch.logical_and(val > 0, first_member > thresh), val, 0.0)
    #ans += torch.where(val > 0, val, 0.0)

    curr_ens_above_thresh = torch.zeros_like(num_ens_above_thresh)
    for i in range(n - 1):
        x0 = torch.index_select(
            pred, dim, torch.tensor([i], device=device, dtype=torch.int32)
        ).squeeze(dim)
        x1 = torch.index_select(
            pred, dim, torch.tensor([i + 1], device=device, dtype=torch.int32)
        ).squeeze(dim)
        curr_ens_above_thresh += (x0 > thresh).long()

        cdf = curr_ens_above_thresh / num_ens_above_thresh
        assert torch.where(num_ens_above_thresh > 0, cdf, 0).max() <= 1
        
        # a. case y < x0
        val = (x1 - x0) * (cdf - 1) ** 2
        mask = obs < x0
        ans += torch.where(torch.logical_and(mask, curr_ens_above_thresh>0), val, 0.0)

        # b. case x0 <= y <= x1
        val = (obs - x0) * cdf**2 + (x1 - obs) * (cdf - 1) ** 2
        mask = (obs >= x0) & (obs <= x1)
        ans += torch.where(torch.logical_and(mask, curr_ens_above_thresh>0), val, 0.0)

        # c. case x1 < y
        mask = obs > x1
        val = (x1 - x0) * cdf**2
        ans += torch.where(torch.logical_and(mask, curr_ens_above_thresh>0), val, 0.0)

    # dx [F(x) - H(x-y)]^2 = dx [1 - 0]^2 = dx
    last_member = torch.index_select(
        pred, dim, torch.tensor([n - 1], device=device, dtype=torch.int32)
    ).squeeze(dim)
    val = obs - last_member
    #ans += torch.where(torch.logical_and(val > 0, last_member > thresh), val, 0.0)
    ans += torch.where(val > 0, val, 0.0)
    ans = torch.where(num_ens_above_thresh > 1, ans, torch.abs(pred.max(dim=dim).values - obs))
    ans = torch.where(obs_above_thresh, ans, torch.nan)
    return ans



def calculate_owcrps_naive(ensemble, observed, threshold):
    """
    Naively calculates owCRPS by iterating over each grid cell.  This is very slow.

    The _owcrps method uses the CDF based calculation of owCRPS.  

    This should result in the same output (allclose) as _owcrps sort-based method above"
    """
    print("Calculating owCRPS")
    threshold_gpu = torch.tensor(threshold, device=ensemble.device)
    start = default_timer()
    owcrps = torch.zeros_like(observed) * torch.nan
    lat, lon = observed.shape
    for i in range(60,720-60): #range(lat-1):
        for j in range(lon):
            if observed[i, j] > threshold_gpu[i, j]:  
                tester_timer = default_timer()
                curr_ensemble = ensemble[i, :, j]
                if (curr_ensemble > threshold_gpu[i, j]).sum() >= 1:
                    curr_ensemble = curr_ensemble[curr_ensemble > threshold_gpu[i, j]]
                    owcrps[i:i+1, j:j+1] = crps(curr_ensemble[None, :, None], observed[i:i+1, j:j+1], method='sort',
                                            dim=1)
                else:
                    owcrps[i, j] = torch.abs(observed[i, j] - curr_ensemble.max())
                #print(default_timer() - tester_timer)

    print(f"Time taken to calculate owCRPS is {default_timer() - start}")
    
    return owcrps.cpu()

def calculate_owcrps(ensemble, observed, thresh):
    output = torch.zeros_like(observed)
    for start_idx in range(0, 720, 90):
        curr_crps = _owcrps(ensemble[start_idx:start_idx +90],
                                         observed[start_idx: start_idx + 90], thresh[start_idx: start_idx + 90],
                                         dim=1)
        output[start_idx : start_idx + 90] = curr_crps
        torch.cuda.empty_cache()
    return output

def calculate_in_chunks_of_90(ensemble, observed):
    output = torch.zeros_like(observed)
    for start_idx in range(0, 720, 90):
        curr_crps = crps(ensemble[start_idx:start_idx +90],
                                         observed[start_idx: start_idx + 90], method='sort',
                                         dim=1)
        output[start_idx : start_idx + 90] = curr_crps
        torch.cuda.empty_cache()
    return output


def load_percentile_threshold(initial_time, lead_time, percentile):
    assert percentile in [95, 99], "Percentile must be 95th or 99th percentile"
    valid_time = initial_time + lead_time
    threshold = xr.open_zarr(f"/pscratch/sd/a/amahesh/hens/thresholds/t2m_percentile{percentile}_{valid_time.month:02d}_{valid_time.hour:02d}/")
    threshold = threshold.rename({
        'VAR_2T' : 't2m',
        'latitude' : 'lat',
        'longitude' : 'lon',
    })
    return threshold['t2m'].values

def load_observed(initial_time, lead_time):
    valid_time = initial_time + lead_time
    true = xr.open_dataset("/pscratch/sd/p/pharring/74var-6hourly/staging/2023.h5", mode='r')
    dummy = xr.open_dataset("/dvs_ro/cfs/cdirs/m1517/cascade/amahesh/hens/HENS_summer23_20230814T000000/ensemble_out_00001_2023-08-14-00-00-00.nc",
                       group='global')

    true = true.rename({'phony_dim_0' : 'time',
             'phony_dim_2' : 'lat', 
             'phony_dim_3' : 'lon',
             'phony_dim_1' : 'channel'})
    true['lat'] = dummy['lat']
    true['lon'] = dummy['lon']
    true['channel'] = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "2d", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000"]
    true['time'] = pd.date_range("2023-01-01", periods=1460, freq='6H')
    return true['fields'].sel(time=valid_time, channel='t2m').load().values

def save_statistic(statistic, reduced_forecast, initial_time_idx, lead_time_idx, fout):
    start = default_timer()
    extended_shape = [92, 12] + list(reduced_forecast.shape)
    if f"{args.variable}_{statistic}" not in fout:
        ds = fout.create_dataset(f"{args.variable}_{statistic}", shape=extended_shape)
    else:
        ds = fout[f"{args.variable}_{statistic}"]
    ds[initial_time_idx, lead_time_idx] = reduced_forecast
    print(f"Time taken to save {statistic} is {default_timer() - start}")

def load_forecast(ensemble_size, variable, initial_time, lead_time_idx, bootstrap):
    if ensemble_size == 7424 or bootstrap:
        print("Loading huge ensemble")
        fin = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/{variable}_{initial_time:%Y%m%d}_lead-04-07-10.h5", 'r')
        assert fin[args.variable].shape[0] == 1, "Expected 1 initial time per file"
        assert fin[args.variable].shape == (1, 12, 721, 7424, 1440), "Expected shape of (1, 12, 721, 7424, 1440) as the dimension shape"
        
        start = default_timer()
        forecast = fin[args.variable][0, lead_time_idx, :, :, :]
        print("Read time is ", default_timer() - start)
        fin.close()
    else:
        print("Loading 58 member ensemble from time collection")
        lead_time_subset = [16,17,18,19,28,29,30,31,40,41,42,43]
        ds = xr.open_zarr(f"/pscratch/sd/a/amahesh/hens/time_collection/bred_29multicheckpoint_pert0p35_k1i3_500km_oppositepert_detfix_nodpr_timeevolve_hemisphererescale_target48_20minus20_newseed_repeat_qperturbfix_rankhist_qmin0_summer2023/{initial_time:%Y-%m-%d}T00:00:00/ensemble.zarr")
        forecast = ds['t2m'].isel(time=lead_time_subset[lead_time_idx])
        forecast = forecast.transpose('lat', 'ensemble', 'lon').values
    return forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Stats on HENS output files")
    parser.add_argument("--variable", type=str, required=True, help="Variable to calculate stats on")
    parser.add_argument("--slurm_array_id", type=int, required=True, help="Slurm array id")
    parser.add_argument("--slurm_array_size", type=int, required=True, help="Slurm array size")
    parser.add_argument("--percentile", type=int, required=True, help="Percentile to calculate stats on")
    parser.add_argument("--bootstrap", action='store_true', help="Use the bootstrapped ensemble")
    args = parser.parse_args()

    assert args.variable == 't2m', "This script is only defined for t2m"
    local_rank = int(os.environ.get('SLURM_LOCALID'))

    np.random.seed(0)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    device_count = torch.cuda.device_count()
    print(f"Rank: {rank}, Size: {size}, Local rank: {local_rank}, device count: {device_count}")

    dates = pd.date_range(start="2023-06-01", end="2023-08-31", freq="D")

    time_subset = np.asarray([16,17,18,19,28,29,30,31,40,41,42,43]) * pd.Timedelta('6H')

    # Data dimensions are (initial_time, lead_time, lat, ensemble, lon)

    for lead_time_idx, lead_time in enumerate(time_subset[:size]):
        if lead_time_idx % size != rank:
            continue
        for initial_time_idx, initial_time in enumerate(dates):
            if initial_time_idx % args.slurm_array_size != args.slurm_array_id:
                continue
            
            thresh = load_percentile_threshold(initial_time, lead_time, args.percentile)
            true = load_observed(initial_time, lead_time)
            bootstrap_str = "yes" if args.bootstrap else "no"
            fout = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/{bootstrap_str}bootstrap_percentile_and_crps/percentile{args.percentile}_{args.variable}_{initial_time:%Y%m%d}_reduced-lead-04-07-10.h5", 'a', driver='mpio', comm=comm)
            
            #Calculate stats on a bootstrapped ensemble
            owcrps_bootstrap_mean, twcrps_bootstrap_mean, crps_bootstrap_mean = [], [], []
            extreme_bootstrap_mean, ens_means, ens_stds = [], [], []
            for ensemble_size in [7424,58]:
                owcrps_bootstrap, twcrps_bootstrap, crps_bootstrap= [], [], []
                extreme_forecasts = []
                start = default_timer()
                forecast = load_forecast(ensemble_size, args.variable, initial_time, lead_time_idx, bootstrap=args.bootstrap)
                ens_means.append(forecast.mean(axis=1))
                ens_stds.append(forecast.std(axis=1))
                trials = 100 if args.bootstrap else 1
                for trial in range(trials):
                    #Clear the ensemble to save memory
                    ensemble = None

                    if args.bootstrap:
                        print("Starting resampling")
                        ensemble = forecast[:, np.random.choice(7424, ensemble_size, replace=True)]
                    else:
                        print("Not resampling")
                        ensemble = forecast
                    
                    assert ensemble.shape == (721, ensemble_size, 1440), f"Expected shape of (721, (ensemble_size), 1440) as the forecast shape.  Got {forecast.shape}"

                    percentile = args.percentile
                    #Calculate the number of ensemble members that exceed the percentile threshold
                    extreme_forecast_probability = (ensemble > thresh[:, None]).mean(axis=1)
                    extreme_forecasts.append(extreme_forecast_probability)

                    #Calculate twCRPS
                    start_twcrps = default_timer()
                    thresh_ensemble = np.where(ensemble < thresh[:, None], thresh[:, None], ensemble).astype(ensemble.dtype)
                    thresh_true = np.where(true < thresh, thresh, true)
                    thresh_ensemble = torch.tensor(thresh_ensemble, device=f"cuda:{local_rank}")
                    thresh_true = torch.tensor(thresh_true, device=f"cuda:{local_rank}")
                    #trial_twcrps = crps(thresh_ensemble, thresh_true, method='sort')
                    trial_twcrps = calculate_in_chunks_of_90(thresh_ensemble, thresh_true)
                    twcrps_bootstrap.append(trial_twcrps.cpu().numpy())
                    print("calculated twcrps in ", default_timer() - start_twcrps)
                    thresh_ensemble, thresh_true = None, None
                    torch.cuda.empty_cache()

                    #Calculate CRPS
                    ensemble_gpu = torch.tensor(ensemble, device=f"cuda:{local_rank}")
                    true_gpu = torch.tensor(true, device=f"cuda:{local_rank}")
                    #trial_crps = crps(ensemble, observed, method='sort')
                    trial_crps = calculate_in_chunks_of_90(ensemble_gpu, true_gpu)
                    crps_bootstrap.append(trial_crps.cpu().numpy())

                    #Calculate owCRPS
                    start_owcrps = default_timer()
                    thresh_gpu = torch.tensor(thresh, device=f"cuda:{local_rank}")
                    trial_owcrps = calculate_owcrps_naive(ensemble_gpu, true_gpu, thresh_gpu)
                    owcrps_bootstrap.append(trial_owcrps.cpu().numpy())
                    print("calculated owcrps in ", default_timer() - start_owcrps)
                    ensemble_gpu, true_gpu = None, None
                    torch.cuda.empty_cache()

                print(f"Time taken to bootstrap ensemble size {ensemble_size} is {default_timer() - start}")
                
                owcrps_bootstrap = np.stack(owcrps_bootstrap, axis=0)
                twcrps_bootstrap = np.stack(twcrps_bootstrap, axis=0)
                crps_bootstrap = np.stack(crps_bootstrap, axis=0)
                extreme_forecasts = np.stack(extreme_forecasts, axis=0)

                owcrps_bootstrap_mean.append(owcrps_bootstrap.mean(axis=0))
                twcrps_bootstrap_mean.append(twcrps_bootstrap.mean(axis=0))
                crps_bootstrap_mean.append(crps_bootstrap.mean(axis=0))
                extreme_bootstrap_mean.append(extreme_forecasts.mean(axis=0))

            owcrps_bootstrap_mean = np.stack(owcrps_bootstrap_mean, axis=0)
            twcrps_bootstrap_mean = np.stack(twcrps_bootstrap_mean, axis=0)
            crps_bootstrap_mean = np.stack(crps_bootstrap_mean, axis=0)
            extreme_bootstrap_mean = np.stack(extreme_bootstrap_mean, axis=0)
            ens_means = np.stack(ens_means, axis=0)
            ens_stds = np.stack(ens_stds, axis=0)

            save_statistic("owcrps", owcrps_bootstrap_mean, initial_time_idx,lead_time_idx, fout)
            save_statistic("twcrps", twcrps_bootstrap_mean,  initial_time_idx,lead_time_idx, fout)
            save_statistic("crps", crps_bootstrap_mean, initial_time_idx, lead_time_idx, fout)
            save_statistic("extreme_forecast", extreme_bootstrap_mean, initial_time_idx, lead_time_idx,fout)
            save_statistic("ensemble_means", ens_means, initial_time_idx, lead_time_idx, fout)
            save_statistic("ensemble_stds", ens_stds, initial_time_idx, lead_time_idx, fout)
            fout.close()

                    

                    


