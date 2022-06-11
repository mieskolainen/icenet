import icenet.algo
import icenet.deep
import icenet.optim
import icenet.tools

from datetime import datetime
import socket

__version__    = '0.0.4'
__release__    = 'alpha'
__date__       = '11/06/2022'
__author__     = 'm.mieskolainen@imperial.ac.uk'
__repository__ = 'github.com/mieskolainen/icenet'
__asciiart__   = \
"""
██  ██████ ███████ ███    ██ ███████ ████████
██ ██      ██      ████   ██ ██         ██
██ ██      █████   ██ ██  ██ █████      ██
██ ██      ██      ██  ██ ██ ██         ██
██  ██████ ███████ ██   ████ ███████    ██
"""

print(__asciiart__)
print(f'{datetime.now()} | hostname: {socket.gethostname()}')
print('')
print(f'MIT license')
print(f' version:    {__version__}')
print(f' release:    {__release__}')
print(f' date:       {__date__}')
print(f' author:     {__author__}')
print(f' repository: {__repository__}')
print(f'')
print(f'')
