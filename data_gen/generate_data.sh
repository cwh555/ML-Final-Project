#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <number_of_batches> <output_directory>"
    echo "Example: $0 5 ./my_dataset"
    exit 1
fi

N=$1
OUTPUT_DIR=$2
SCRIPT="data_gen_src/gendata.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in the current directory."
    exit 1
fi

echo "Starting batch generation..."
echo "Running '$SCRIPT' $N times."
echo "Target Directory: $OUTPUT_DIR"
echo "------------------------------"

mkdir -p "$OUTPUT_DIR"

# Loop N times
for (( i=1; i<=N; i++ ))
do
    echo "Batch $i/$N"
    python3 "$SCRIPT" --output_dir "$OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error occurred in batch $i. Stopping."
        exit 1
    fi
done

echo "------------------------------"
echo "Completed. All files saved to '$OUTPUT_DIR'."
