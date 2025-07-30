# Recursive TensorBoard hyperparam creator
# 
# Example:
# 
# Step 1
# python icefit/iceboard.py \
#    --rootpath figs/zee/config__tune0_EB.yml \
#    --variables evaltag beta sigma tau --tag mytest
# 
# Step 2
# tensorboard --logdir ./tmp/iceboard/mytest/hparam_logs
# 
# Terminate session: pkill tensorboard
# 
# m.mieskolainen@imperial.ac.uk, 2024

import tensorflow as tf # Keep at the top

import os
import argparse
import time
import re
import shutil
from termcolor import cprint
from pathlib import Path

from tensorboard.plugins.hparams import api as hp
from collections import defaultdict

def read_tensorboard_data(tensorboard_file):
    """
    Reads scalar data from a TensorBoard file using tf.data.TFRecordDataset.
    """
    metric_values = {}
    for record in tf.data.TFRecordDataset(tensorboard_file):
        event = tf.compat.v1.Event.FromString(record.numpy())
        for v in event.summary.value:
            if v.HasField('simple_value'):
                metric_values[v.tag] = v.simple_value
    return metric_values

def combine_hparam(rootdir):
    
    # Pattern to match parameter folders (e.g., 'beta_0.1')
    param_pattern = re.compile(r'([^_/]+)_(.+)')

    # Dictionary to hold parameter values for determining ranges
    param_values = defaultdict(set)

    # List to hold all runs with their parameters and TensorBoard file paths
    runs = []

    # Set to collect all metric tags
    all_metric_tags = set()

    # Recursively traverse the directory structure
    for dirpath, dirnames, filenames in os.walk(rootdir):
        # If current directory is a leaf directory (no subdirectories)
        if not dirnames:
            # If there is exactly one file in filenames
            if len(filenames) == 1:
                tensorboard_file = os.path.join(dirpath, filenames[0])

                # Extract parameter names and values from the directory path
                relative_path = os.path.relpath(dirpath, rootdir)
                path_parts = relative_path.split(os.sep)

                hparams = {}
                for part in path_parts:
                    match = param_pattern.match(part)
                    if match:
                        param_name = match.group(1)
                        param_value = match.group(2)
                        # Keep the original string value
                        hparams[param_name] = param_value
                        param_values[param_name].add(param_value)

                # Read scalar data from the existing TensorBoard file
                metric_values = read_tensorboard_data(tensorboard_file)
                all_metric_tags.update(metric_values.keys())

                # Add the run to the list
                runs.append({
                    'hparams': hparams,
                    'metric_values': metric_values
                })
            else:
                print(f"Warning: Directory '{dirpath}' does not contain exactly one file.")
                # Optionally handle multiple files here

    if not runs:
        print("No runs found. Make sure your directory structure and TensorBoard files exist.")
        return

    # Define hyperparameters and determine their types and ranges
    hparams_list = []
    for param_name, values in param_values.items():
        values_list = list(values)
        # Attempt to convert all values to floats
        all_numeric = True
        numeric_values = []
        for v in values_list:
            try:
                numeric_values.append(float(v))
            except ValueError:
                all_numeric = False
                break
        if all_numeric:
            # Determine if all values are integers
            if all(float(v).is_integer() for v in values_list):
                # Convert to integers
                numeric_values = [int(float(v)) for v in values_list]
                min_value = min(numeric_values)
                max_value = max(numeric_values)
                hparam = hp.HParam(param_name, hp.IntInterval(min_value, max_value))
            else:
                # Keep as floats
                min_value = min(numeric_values)
                max_value = max(numeric_values)
                hparam = hp.HParam(param_name, hp.RealInterval(min_value, max_value))
        else:
            # Treat as categorical parameter
            hparam = hp.HParam(param_name, hp.Discrete(sorted(values_list)))
        hparams_list.append(hparam)

    # Define the metrics based on collected tags
    metrics = [hp.Metric(tag, display_name=tag.capitalize()) for tag in all_metric_tags]

    # Create a top-level log directory for hparams
    hparam_log_dir = os.path.join(rootdir, 'hparam_logs')

    # Write the hparams configuration (only once)
    with tf.summary.create_file_writer(hparam_log_dir).as_default():
        hp.hparams_config(
            hparams=hparams_list,
            metrics=metrics,
        )

    # Process each run
    for idx, run in enumerate(runs):
        hparams = run['hparams']
        metric_values = run['metric_values']

        # Create a unique run name for each hyperparameter combination
        run_name_parts = [f"{k}_{v}" for k, v in hparams.items()]
        run_name = '_'.join(run_name_parts)

        # Create a run-specific log directory
        run_log_dir = os.path.join(hparam_log_dir, run_name)

        # Start a new run in TensorBoard
        with tf.summary.create_file_writer(run_log_dir).as_default():
            # Convert hyperparameter values to appropriate types
            hparams_converted = {}
            for k, v in hparams.items():
                try:
                    v_converted = float(v)
                    if v_converted.is_integer():
                        v_converted = int(v_converted)
                    hparams_converted[k] = v_converted
                except ValueError:
                    hparams_converted[k] = v  # Keep as string
            hp.hparams(hparams_converted)  # Record the hyperparameters

            for tag, value in metric_values.items():
                tf.summary.scalar(tag, value, step=0)  # Use step=0 for initial logging

        # Print the hyperparameters and metrics to the screen
        print(f"Processed run: {run_name}")
        print("Hyperparameters:")
        for key, value in hparams_converted.items():
            print(f"  {key}: {value}")
        print("Metrics:")
        for tag, value in metric_values.items():
            print(f"  {tag}: {value}")
        print("-" * 40)

    print(f"All runs have been processed. TensorBoard log directory: {hparam_log_dir}")

    return hparam_log_dir

def create_symlink(src, dest):
    """
    Create symbolic link from src to dest using absolute paths.
    If the link exists, it will be removed and recreated.
    """
    # Convert src and dest to absolute paths
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    
    dest_path = Path(dest)
    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # If the symbolic link already exists, remove it
    if dest_path.exists() or dest_path.is_symlink():
        dest_path.unlink()
    # Create the symbolic link
    os.symlink(src, dest)
    cprint(f"Created symlink:", 'yellow')
    cprint(f"{dest} --> ", 'green') 
    cprint(f"{src}", 'red')
    print("")

def extract_variable_values(current_path_parts, variables):
    """
    Extract variable values from the directory parts.
    Returns a dictionary of variable names and their values if all variables are found, else None.
    """
    var_values = {}
    # Reverse the parts to start from the deepest directory
    for part in reversed(current_path_parts):
        # Split the directory name by '__' and '_'
        tokens = re.split(r'__|_', part)
        # Iterate over tokens to find variables
        for i, token in enumerate(tokens):
            for var in variables:
                if token == var and i + 1 < len(tokens):
                    value = tokens[i + 1]
                    var_values[var] = f"{var}_{value}"
        # Check if all variables have been found
        if len(var_values) == len(variables):
            return var_values
    return None

def process_folders(root_path, variables, output_base, max_files=None):
    """
    Process folders to create symbolic links under the output_base directory with the desired structure.
    """
    files_processed = 0
    for dirpath, _, filenames in os.walk(root_path):
        current_path = Path(dirpath)
        current_path_parts = current_path.parts

        # Extract variable values from the directory path
        var_values = extract_variable_values(current_path_parts, variables)

        if var_values:
            # Ensure the variables are in the same order as specified
            relative_path_parts = [var_values[var] for var in variables]

            # Filter for TensorBoard event files
            for file in filenames:
                if file.startswith("events.out.tfevents"):
                    src_path = os.path.join(dirpath, file)
                    dest_path = os.path.join(output_base, *relative_path_parts, file)

                    # Convert src_path and dest_path to absolute paths
                    src_path = os.path.abspath(src_path)
                    dest_path = os.path.abspath(dest_path)

                    create_symlink(src_path, dest_path)

                    files_processed += 1

                    # Check if we've reached the maximum number of files
                    if max_files is not None and files_processed >= max_files:
                        print(f"Reached maximum number of files to process: {max_files}")
                        return
        else:
            continue

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create symbolic links for TensorBoard logs based on a folder structure.")
    parser.add_argument(
        "--rootpath",
        type=str,
        required=True,
        help="Root directory where the actual TensorBoard log files are located."
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs='+',
        required=True,
        help="Variable names to detect in the folder structure (e.g., beta sigma tau)."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of TensorBoard log files to process."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag name to use for the output directory instead of a timestamp."
    )
    args = parser.parse_args()

    # Generate output directory
    if not args.tag:
        args.tag = timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    output_base = os.path.join("./tmp/iceboard", args.tag)

    # Remove the old directory if it exists (force and silent)
    if os.path.exists(output_base):
        shutil.rmtree(output_base, ignore_errors=True)

    # Convert output_base to absolute path
    output_base = os.path.abspath(output_base)

    # Process folders and create symbolic links
    process_folders(args.rootpath, args.variables, output_base, max_files=args.max_files)
    cprint(f"Symbolic links created to {output_base}.", 'green')

    # Combine to HPARAM
    hparam_logdir = combine_hparam(output_base)
    
    print(f"Run tensorboard with:")
    cprint(f"tensorboard --logdir {hparam_logdir} (--port 6000)", 'green')
