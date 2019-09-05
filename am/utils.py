import logging
from functools import wraps
from pathlib import Path
from shutil import rmtree
from time import time

logger = logging.getLogger('am-segm')


def clean_dir(path):
    logger.info(f'Cleaning up {path} directory')
    rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True)


def time_it(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        minutes, seconds = divmod(time() - start, 60)
        logger.info(f"Function '{func.__name__}' running time: {minutes:.0f}m {seconds:.0f}s")
        return res

    return wrapper
