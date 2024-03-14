from datetime import datetime
import socket
import os
import psutil

__version__    = '0.0.9.9'
__release__    = 'alpha'
__date__       = '14/03/2024'
__author__     = 'm.mieskolainen@imperial.ac.uk'
__repository__ = 'github.com/mieskolainen/icenet'
__asciiart__   = \
"""
ICENET
"""

total = psutil.virtual_memory()[0]/1024**3
free  = psutil.virtual_memory()[1]/1024**3

print(f'{__asciiart__} version: {__version__} | date: {__date__} | author: {__author__} | repository: {__repository__}')
print(f' {datetime.now()} | hostname: {socket.gethostname()} | CPU cores: {os.cpu_count()} | RAM: {total:0.1f} ({free:0.1f}) GB')
print('')
