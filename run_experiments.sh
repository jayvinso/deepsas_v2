#!/bin/bash

# Define input and output directories
SUBSAMPLE_DIR="/path/to/your/subsamples"  # Update this to your subsample directory
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
