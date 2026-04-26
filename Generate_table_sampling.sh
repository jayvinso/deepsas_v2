

#!/bin/bash

# Loop from 1 to 28

cd /bmbl_data/ahmed/sencell

for i in {1..28}
do
    # Define the file path dynamically
    FILE_PATH="/bmbl_data/ahmed/eye_atlas/subsamples/subsample_${i}.h5ad"
    
    # Define the experiment name dynamically
    EXP_NAME="Data_${i}_022425"
    
    # Run your command (replace `your_command` with the actual command you want to execute)
    uv run python -u generate_3tables.py --input_data_count "$FILE_PATH" --output_dir ./outputs --exp_name "$FILE_PATH" --device_index 4 --timestamp '022425' --retrain

    # Print (optional, for debugging)
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done


