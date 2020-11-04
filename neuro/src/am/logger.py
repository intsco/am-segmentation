import logging


def init_logger(level=logging.INFO):
    logger = logging.getLogger('am-segm')
    if len(logger.handlers) < 1:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
