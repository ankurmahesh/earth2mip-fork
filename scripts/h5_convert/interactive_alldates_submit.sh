
export HDF5_USE_FILE_LOCKING=FALSE

srun -N 4 -u --mpi=pmi2 --module=none -n 736 -c 1 --cpu_bind=threads shifter --image=nersc/pytorch:ngc-22.02-v0 python h5_convert.py --variable wind_speed10m


#srun -N 4 -u --mpi=pmi2 --module=none -n 16 -c 8 --cpu_bind=cores shifter --image=nersc/pytorch:ngc-22.02-v0 bash -c "export HDF5_USE_FILE_LOCKING=FALSE; python h5_convert.py --variable wind_speed10m"

#srun -N 1 -u -n 1 -c 4 --cpu_bind=cores python h5_convert.py
