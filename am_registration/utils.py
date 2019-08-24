import logging


def init_logger():
    logger = logging.getLogger('am-reg')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def min_max(a):
    return a.min(), a.max()


def cut_patch(image, y_offset=0, x_offset=0, patch=1000):
    return image[y_offset:y_offset+patch, x_offset:x_offset+patch]
