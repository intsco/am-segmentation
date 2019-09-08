import os
from pathlib import Path
import falcon

from api.resources import AblationMask, SegmentationTask, SegmentationTaskCollection
from api.task_manager import TaskManager
from api.utils import logger

DEFAULT_DATA_PATH = './data/tasks'


def create_app(data_path):
    task_manager = TaskManager(data_path)
    api = falcon.API()
    # api.add_route('/masks', am_masks.AblationMaskCollection(mask_store))
    api.add_route('/masks/{task_id}', AblationMask(task_manager))
    api.add_route('/tasks', SegmentationTaskCollection(task_manager))
    api.add_route('/tasks/{task_id}', SegmentationTask(task_manager))
    return api


def get_app():
    data_path = Path(os.environ.get('AM_DATA_PATH', DEFAULT_DATA_PATH))
    return create_app(data_path)


if __name__ == '__main__':
    logger.info('Creating app...')
    app = get_app()
    from wsgiref import simple_server
    httpd = simple_server.make_server('127.0.0.1', 8000, app)
    logger.info('Running debug server...')
    httpd.serve_forever()
