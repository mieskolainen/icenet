# Sun Grid Enginer (qsub) submission helper tool
#
# m.mieskolainen@imperial.ac.uk, 2023

import os
import yaml
import argparse

def parse_job(p, args):

    cmd = f"qsub"

    # Queue
    if p['q'] is not None:
        cmd += f" -q {p['q']}"

    # Array job
    if p['t'] is not None:
        cmd += f" -t {p['t']}"

    # Hard limit on memory
    if p['h_vmem'] is not None:
        cmd += f" -l h_vmem={p['h_vmem']}"

    # Runtime
    if p['h_rt'] is not None:
        cmd += f" -l h_rt={p['h_rt']}"

    # Multicore
    if p['pe'] is not None:
        cmd += f" -pe {p['pe']}"

    # Shell
    if p['S'] is not None:
        cmd += f" -S {p['S']}"

    # Email
    if p['m'] is not None:
        cmd += f" -M {args.email} -m {p['m']}"

    # --------------------------
    # Add binary file support
    cmd += f" -b y"

    # Resubmit if fails
    cmd += f" -r y"
    
    # Hard requirements
    cmd += f" -hard"
    
    # Job name
    cmd += f" -N {args.job}"
    
    # Output logging
    cmd += f" -o {args.output}"
    cmd += f" -e {args.output}"
    # --------------------------

    # Finally, command to execute
    if p['cwd'] is not None:
        cmd += f" -cwd {p['cwd']}"

    return cmd

def main():

    parser = argparse.ArgumentParser(
                    prog='iceqsub',
                    description='Sun Grid Engine qsub steering program')

    # Argument parser
    parser.add_argument('-c', '--config', default='configs/qsub/config.yml')
    parser.add_argument('-j', '--job',    default='hello_world')
    parser.add_argument('-r', '--run',    action='store_true')  # on/off flag
    parser.add_argument('-M', '--email',  default='m.mieskolainen@imperial.ac.uk')
    
    parser.add_argument('-o', '--output', default='$HOME')  # std output
    parser.add_argument('-e', '--error',  default='$HOME')  # error output
    
    args = parser.parse_args()
    print(args)

    # Read YAML-config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Pick particular job
    if args.job in config['jobs']:
        p   = config['jobs'][args.job]
        cmd = parse_job(p=p, args=args)
    else:
        raise Exception(f"Job '{args.job}' not found in config.yml")
    
    print('Command to execute (use iceqsub --run to execute):')
    print('')
    print(f'{cmd}')
    print('')
    
    if args.run:
        os.system(cmd)

if __name__ == "__main__":
    main()
