import io
import logging


logger = logging.getLogger('am-segm')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def save_status(path, status):
    logger.info(f'Saving status {status} to {path}')
    with io.open(path, 'w') as f:
        f.write(status)
