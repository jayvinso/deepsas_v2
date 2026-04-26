# Run DeepSAS by subsampling your large anndata object
In this branch we show how to do subsampling for your anndata object, run DeepSAS, and generate tables/results for each subsampled object


### Usage

1. susampling your anndata object using script subsampling.py using the following command

```bash
uv run python subsampling.py --input your_data.h5ad --subsample_size 30000 --output ./subsamples
```
3. Once you created subsamples you can use 'run_experiments.sh' to loop through each subsample and run DeepSAS

4. In your bash enviroment 

```bash

#!/bin/bash

# Define input and output directories
SUBSAMPLE_DIR="/bmbl_data/ahmed/eye_atlas/subsamples"  # Update this to your subsample directory
OUTPUT_DIR="./outputs"
DEVICE_INDEX=4  # Set your GPU device index

# Loop over all subsample files dynamically
for FILE_PATH in "$SUBSAMPLE_DIR"/subsample_*.h5ad; do
    # Extract the iteration number from the filename (e.g., "subsample_1.h5ad" -> "1")
    FILE_NAME=$(basename "$FILE_PATH")  # Extract filename
    ITER_NUM=$(echo "$FILE_NAME" | grep -oP '\d+')  # Extract numeric part

    # Define the experiment name dynamically
    EXP_NAME="Data_${ITER_NUM}_013025"

    # Run the DeepSAS command for each subsample
    uv run python -u deepsas_v1.py \
        --input_data_count "$FILE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "$EXP_NAME" \
        --device_index "$DEVICE_INDEX" \
        --retrain

    # Print for debugging
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done

```
once you adjust SUBSAMPLE_DIR, OUTPUT_DIR, and DEVICE_INDEX you can save the script and run the following command

```bash
nohup bash run_experiments.sh

```


To generate 3 table of SnGs run 'Generate_table_sampling.sh' :
```bash
for i in {1..28}
do
    # Define the file path dynamically
    FILE_PATH="/path/to/your/subsamples/subsample_${i}.h5ad"
    
    # Define the experiment name dynamically
    EXP_NAME="Data_${i}_022425"
    
    # Run your command (replace `your_command` with the actual command you want to execute)
    uv run python -u generate_3tables.py --input_data_count "$FILE_PATH" --output_dir ./outputs --exp_name "$FILE_PATH" --device_index 4 --retrain

    # Print (optional, for debugging)
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done
```
This code will generate a folder for each subsample and each folder contail sncG and sncC results
The `generate_3tables.py` needs two inputs: .h5ad file and DeepSAS output. you can use `--input_data_count` and `--exp_name` to load your .h5ad and DeepSAS output respectively.

For the visualization and downstream analysis of SnCs and SnGs, please follow the tutorial in the [`tutorial.ipynb`](./tutorial.ipynb).


#### Important Arguments

- `--output_dir`: Directory to store output files and results.
- `--exp_name`: Descriptive name for the experiment.
- `--device_index`: GPU device index if CUDA is available.
