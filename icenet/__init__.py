from datetime import datetime
import socket
import os
import psutil

__version__    = '0.1.3.1'
__release__    = 'alpha'
__date__       = '19/10/2024'
__author__     = 'm.mieskolainen@imperial.ac.uk'
__repository__ = 'github.com/mieskolainen/icenet'
__asciiart__   = \
"""
ICENET
"""

print(f'{__asciiart__} version: {__version__} | date: {__date__} | author: {__author__}')
print(f' {datetime.now()} | hostname: {socket.gethostname()} | CPU cores: {os.cpu_count()} | RAM: {psutil.virtual_memory()[0]/1024**3:0.1f} (free {psutil.virtual_memory()[1]/1024**3:0.1f}) GB')
print(f' In talks and publications, please cite: <{__repository__}> (MIT license)')
print('')

from icenet.tools import icelogger
icelogger.set_global_log_file('initial.log')
LOGGER = icelogger.get_logger()

from icenet.tools.iceprint import iceprint
print = iceprint
