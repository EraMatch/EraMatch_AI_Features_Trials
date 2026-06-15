#!/bin/bash

# Define the output file name
OUTPUT_FILE="merged_triplets.jsonl"

# Check if any matching files exist
if ls triplets_part_*.jsonl >/dev/null 2>&1; then
    echo "Merging files..."
    # Concatenate all matching files into the output file
    cat triplets_part_*.jsonl > "$OUTPUT_FILE"
    echo "Success! Combined files into '$OUTPUT_FILE'."
else
    echo "Error: No files matching 'triplets_part_*.jsonl' were found in the current directory."
    exit 1
fi