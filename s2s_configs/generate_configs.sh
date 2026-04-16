#!/bin/bash                                                             

# Load the contents of sample_config into a variable                       
sample_config=$(<sample_config)                                         
                                                                        
# Define the output directory                                           
output_dir="./configs"                                                  
                                                                        
# Create the output directory if it doesn't exist                       
mkdir -p "$output_dir"                                                  
                                                                        
# Function to generate config file for a given date                     
generate_config() {                                                     
    local new_date="$1"                                                 
    local formatted_date=$(date -u -d "$new_date" "+%Y%m%dT000000")        
    local start_formatted_date=$(date -u -d "$new_date" "+%Y-%m-%d 00:00:00")
    local output_path="/pscratch/sd/y/ypeings/HENS_data/s2s_$formatted_date/"
    local config_file="config_${formatted_date}.json"                   
    local new_config=$(echo "$sample_config" | sed -e "s/\"start_time\": \"2023-06-01 00:00:00\"/\"start_time\": \"$start_formatted_date\"/" -e "s#/pscratch/sd/a/amahesh/hens/HENS_summer23_20230601#$output_path#")
    echo "$new_config" > "$output_dir/$config_file"                     
}                                                                       
                                                                        
# Array of specific dates                                               
dates=("01-02" "01-09" "01-16" "01-23" "01-30"
       "02-06" "02-13" "02-20" "02-27" "03-06"
       "10-03" "10-10" "10-17" "10-24" "10-31"
       "11-07" "11-14" "11-21" "11-28" "12-05"
       "12-12" "12-19" "12-26")                 
                                                                        
# Loop through each year from 2018 to 2023                              
#for YEAR in {2018..2022}; do                                             
#
#    for current_date in "${dates[@]}"; do                                
#        generate_config "${YEAR}-${current_date}T00:00:00Z"              
#    done                                                                
#done                                                                
#
#dates=("01-02" "01-09" "01-16" "01-23" "01-30"
#    "02-06" "02-13" "02-20" "02-27" "03-06")
#
#YEAR=2023
#for current_date in "${dates[@]}"; do                                
#    generate_config "${YEAR}-${current_date}T00:00:00Z"              
#done                                                                


# Loop through each year from 2004 to 2017                            
for YEAR in {2004..2017}; do                                             

    for current_date in "${dates[@]}"; do                                
        generate_config "${YEAR}-${current_date}T00:00:00Z"              
    done                                                                
done                                                                
