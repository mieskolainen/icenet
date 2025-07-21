# Colored printing
#
# m.mieskolainen@imperial.ac.uk, 2025

import inspect
from datetime import datetime

from functools import wraps

from icenet import LOGGER

# ANSI escape codes for coloring
COLORS = {
    'white': '\033[97m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'reset': '\033[0m'
}

def icelog(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f'Calling function {func.__name__} with args: {args}, kwargs: {kwargs}')
            result = func(*args, **kwargs)
            logger.info(f'Function {func.__name__} returned {result}')
            return result
        return wrapper
    return decorator

def iceprint(message, color='white', file=None, end=None):
    
    # Get the caller frame
    caller_frame = inspect.currentframe().f_back
    # Get the caller function name
    func_name = caller_frame.f_code.co_name
    # Get the caller class name if available
    class_name = caller_frame.f_globals.get('__name__', '')
    
    # Form the prefix
    current_time = datetime.now().strftime('[%H:%M:%S]')
    prefix = f"{current_time} {class_name}.{func_name}"
    
    # Print the message with color
    color_code = COLORS.get(color, COLORS['white'])
    reset_code = COLORS['reset']
    
    if file is None:
        
        if type(message) is str and (message == '' or message == '\n'):
            print(message, end=end)
            
        # String
        elif type(message) is str:
            print(f"{color_code}{prefix}: {message}{reset_code}", end=end)
            
            LOGGER.info(f'{prefix}: {message}')
            
        # Print objects on a separate line
        else:
            print(f"{prefix}:")
            print(f"{color_code}{message}{reset_code}", end=end)
        
            LOGGER.info(message)
        
    else:
        print(message, file=file)
