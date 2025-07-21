# Recompress and split parquet (pandas) file into multiple files
#
# Example:
#   python icefit/rezip.py input_data.parquet output_file_name \
#       --max_rows_per_file 500000 --compression brotli --compression_level 9
# 
# m.mieskolainen@imperial.ac.uk, 2025

import pandas as pd
import os
import argparse

def split_parquet(df, max_rows_per_file):
    
    # Split the DataFrame into chunks based on the maximum number of rows
    num_chunks = len(df) // max_rows_per_file + (1 if len(df) % max_rows_per_file != 0 else 0)
    
    # Yield chunks of DataFrame
    for i in range(num_chunks):
        start_idx = i * max_rows_per_file
        end_idx = start_idx + max_rows_per_file
        yield df.iloc[start_idx:end_idx]

def compress_parquet(df, output_file, compression, compression_level):
    print(f'Compressing data to "{output_file}"')
    print(f'Using compression "{compression}" at level {compression_level}')
    
    # Write the DataFrame directly to a compressed Parquet file
    df.to_parquet(output_file, compression=compression, compression_level=compression_level)
    
    # Get the file sizes
    output_size = os.path.getsize(output_file)
    print(f"Output File Size: {output_size / (1024 ** 2):.2f} MB")
    
    return output_size

def main():
    
    print('rezip (m.mieskolainen@imperial.ac.uk, 2025)')
    
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split and/or compress a Parquet file.")
    
    # Required arguments
    parser.add_argument("input_file",  type=str, help="Path to the input .parquet file")
    parser.add_argument("output_file", type=str, help="Base name for output files")
    
    # Argument for maximum rows per file
    parser.add_argument(
        "--max_rows_per_file", 
        type=int, 
        required=True, 
        help="Maximum number of rows per output file"
    )
    
    # Arguments for compression
    parser.add_argument(
        "--compression", 
        type=str, 
        choices=["gzip", "brotli"], 
        default="brotli", 
        help="Compression algorithm to use (gzip or brotli). Default is brotli."
    )
    parser.add_argument(
        "--compression_level", 
        type=int, 
        default=11, 
        help="Compression level. Default is 11 (max for brotli). Max 9 for gzip."
    )
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Get the input file size
    input_size = os.path.getsize(args.input_file)
    print(f"Input File Size: {input_size / (1024 ** 2):.2f} MB ({args.input_file})")
    
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(args.input_file)
    
    # Split the DataFrame into chunks
    chunk_number = 0
    tot_output_size = 0
    for chunk_df in split_parquet(df, args.max_rows_per_file):
        # Automatically create output file names by appending the chunk number
        output_file = f"{args.output_file}_{chunk_number}.parquet"
        tot_output_size += compress_parquet(chunk_df, output_file, args.compression, args.compression_level)
        chunk_number += 1

    print(f'Total output size: {tot_output_size / (1024 ** 2):.2f} MB')
    print()

if __name__ == "__main__":
    main()
