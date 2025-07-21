#!/bin/bash
# 
# Read in parquet files, and split & re-compress
# 
# Usage: ./parquet_rezip.sh <input_name.parquet> <output_name> <max_rows>
# 
# Example:
# 
# ./tests/parquet_rezip.sh \
#  /vols/cms/pfk18/icenet_files/processed_20Feb2025 \
#  /vols/cms/mmieskol/icenet/actions-stash/input/icezee \
#  1000000
#
# m.mieskolainen@imperial.ac.uk, 2025

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MAX_ROWS="$3"

# Create output dir if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over all parquet files in the input directory
for FILE in "$INPUT_DIR"/*.parquet; do
    
    # Get the base filename (no directory, no extension)
    BASENAME=$(basename "$FILE" .parquet)
    
    # Construct the output path
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}"

    # Call the rezip script
    python icefit/rezip.py "$FILE" "$OUTPUT_FILE" \
        --max_rows_per_file "$MAX_ROWS" \
        --compression brotli \
        --compression_level 9
done

ls "$OUTPUT_DIR" -l -h
