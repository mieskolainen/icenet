# Recompress Pandas parquet file
#
# Example: python icefit/rezip.py input.parquet output.parquet
#
# m.mieskolainen@imperial.ac.uk, 2024

import pandas as pd
import argparse
import os

def compress_parquet(input_file, output_file, compression, compression_level):
    
    print(f'Compressing "{input_file}" to "{output_file}"')
    print(f'Using compression "{compression}" at level {compression_level}')
    
    # Read the existing parquet file and compress
    df = pd.read_parquet(input_file)
    df.to_parquet(output_file, compression=compression, compression_level=compression_level)
    
    # Get the file sizes
    input_size  = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    reduction_percentage = ((input_size - output_size) / input_size) * 100
    
    # Print file sizes and reduction
    print(f"Input File Size:  {input_size / (1024 ** 2):.2f} MB")
    print(f"Output File Size: {output_size / (1024 ** 2):.2f} MB")
    print(f"Size Reduction:   {reduction_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a Parquet file with specified compression algorithm.")
    parser.add_argument("input_file",  type=str, help="Path to the input Parquet file.")
    parser.add_argument("output_file", type=str, help="Path to save the compressed Parquet file.")
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
    args = parser.parse_args()
    
    compress_parquet(args.input_file, args.output_file, args.compression, args.compression_level)
