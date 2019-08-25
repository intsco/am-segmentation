import logging


def init_logger():
    logger = logging.getLogger('am-segm')
    if len(logger.handlers) < 1:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
