import os
from pathlib import Path
import falcon

from api.am_masks import AblationMask, SegmentationTask, SegmentationTaskCollection
from api.segmentator import Segmentator
from utils import logger


def create_app(data_path='./data/tasks'):
    segmentator = Segmentator(data_path)
    api = falcon.API()
    # api.add_route('/masks', am_masks.AblationMaskCollection(mask_store))
    api.add_route('/masks/{task_id}', AblationMask(segmentator))
    api.add_route('/tasks', SegmentationTaskCollection(segmentator))
    api.add_route('/tasks/{task_id}', SegmentationTask(segmentator))
    return api


def get_app():
    data_path = Path(os.environ.get('LOOK_STORAGE_PATH', './data'))
    return create_app(data_path)


if __name__ == '__main__':
    logger.info('Creating app...')
    app = create_app()
    from wsgiref import simple_server
    httpd = simple_server.make_server('127.0.0.1', 8000, app)
    logger.info('Running debug server...')
    httpd.serve_forever()
