#!/bin/bash                                                             
#SBATCH -N 1                                                               
#SBATCH -C 'gpu&hbm80g'                                                    
#SBATCH -q regular                                                      
#SBATCH -J s2s_config                                                   
#SBATCH --mail-user=amahesh@lbl.gov                                     
#SBATCH --mail-type=ALL                                                 
#SBATCH -t 02:30:00                                                     
#SBATCH -A m4416                                                        
#SBATCH -o /pscratch/sd/y/ypeings/HENS_data/logs/%j_%A_%a.out          
#SBATCH --array=0-321  # Adjusted array size for 115 specific dates                             
#SBATCH --mail-type=ARRAY_TASKS                                         
                                                                        
#years=("2018" "2019" "2020" "2021" "2022")                              
#                                                                        
## Initialize an empty array to store all dates                          
#dates=()                                                                
#                                                                        
## Loop through each year and populate the dates array                   
#for YEAR in "${years[@]}"; do                                           
#    dates+=("${YEAR}0102" "${YEAR}0109" "${YEAR}0116" "${YEAR}0123" "${YEAR}0130"
#            "${YEAR}0206" "${YEAR}0213" "${YEAR}0220" "${YEAR}0227" "${YEAR}0306"
#            "${YEAR}1003" "${YEAR}1010" "${YEAR}1017" "${YEAR}1024" "${YEAR}1031"
#            "${YEAR}1107" "${YEAR}1114" "${YEAR}1121" "${YEAR}1128" "${YEAR}1205"
#            "${YEAR}1212" "${YEAR}1219" "${YEAR}1226")
#done
#
#YEAR="2023"
#dates+=("${YEAR}0102" "${YEAR}0109" "${YEAR}0116" "${YEAR}0123" "${YEAR}0130"
#    "${YEAR}0206" "${YEAR}0213" "${YEAR}0220" "${YEAR}0227" "${YEAR}0306")


years=("2004" "2005" "2006" "2007" "2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017")

# Initialize an empty array to store all dates                          
dates=()                                                                
                                                                        
# Loop through each year and populate the dates array                   
for YEAR in "${years[@]}"; do                                           
    dates+=("${YEAR}0102" "${YEAR}0109" "${YEAR}0116" "${YEAR}0123" "${YEAR}0130"
            "${YEAR}0206" "${YEAR}0213" "${YEAR}0220" "${YEAR}0227" "${YEAR}0306"
            "${YEAR}1003" "${YEAR}1010" "${YEAR}1017" "${YEAR}1024" "${YEAR}1031"
            "${YEAR}1107" "${YEAR}1114" "${YEAR}1121" "${YEAR}1128" "${YEAR}1205"
            "${YEAR}1212" "${YEAR}1219" "${YEAR}1226")
done

# Select the current date based on the SLURM_ARRAY_TASK_ID              
current_date="${dates[$SLURM_ARRAY_TASK_ID]}T000000"                      
echo $current_date
                                                                        
# Your existing srun command with the selected date                     
srun -N 1 --ntasks-per-node=2 --cpus-per-task=64 --gpus-per-node=2 -u shifter --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint --module=gpu,nccl-2.18 bash -c "source set_74ch_vars.sh; python -m earth2mip.inference_ensemble s2s_configs/config_$current_date.json"

