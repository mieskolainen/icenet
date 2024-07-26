# Logging tools (take a note that multiprocessing may result in log message collisions)
#
# m.mieskolainen@imperial.ac.uk, 2024

import logging
from functools import wraps

class SingletonLogger:
    _instance = None
    _log_file = 'initial.log'

    @staticmethod
    def get_instance():
        if SingletonLogger._instance is None:
            SingletonLogger()
        return SingletonLogger._instance

    def __init__(self):
        if SingletonLogger._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SingletonLogger._instance = self
            self.logger = logging.getLogger('icelogger')
            self.logger.setLevel(logging.INFO)
            self.set_log_file(SingletonLogger._log_file)

    def set_log_file(self, log_file):
        SingletonLogger._log_file = log_file
        # Remove all old handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            
        logger_blocklist = [
            "fiona",
            "rasterio",
            "matplotlib",
            "PIL",
        ]

        for module in logger_blocklist:
            logging.getLogger(module).setLevel(logging.WARNING)
        
        self.logger.addHandler(handler)

def get_logger():
    return SingletonLogger.get_instance().logger

def set_global_log_file(log_file):
    SingletonLogger.get_instance().set_log_file(log_file)

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
