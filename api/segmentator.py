import io
import mimetypes
import os
import uuid
from pathlib import Path
import redis

from utils import logger, save_status


class Segmentator(object):

    _CHUNK_SIZE_BYTES = 4096

    def __init__(self, storage_path):
        self._storage_path = Path(storage_path)
        self._uuidgen = uuid.uuid4
        self._fopen = io.open
        self._redis_client = redis.Redis()

    def create_task(self, image_stream, image_content_type):
        task_id = str(uuid.uuid4())
        task_path = self._storage_path / task_id
        task_path.mkdir(exist_ok=True)
        ext = mimetypes.guess_extension(image_content_type)
        image_path = task_path / f'image{ext}'
        self._save_image(image_stream, image_path)

        self._redis_client.rpush('task-queue', task_id)
        save_status(task_path / 'status.txt', 'QUEUED')
        return task_id

    def task_status(self, task_id):
        with io.open(self._storage_path / task_id / 'status.txt', 'r') as f:
            return f.read()

    def _save_image(self, image_stream, image_path):
        with io.open(image_path, 'wb') as image_file:
            while True:
                chunk = image_stream.read(self._CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                image_file.write(chunk)

    def read_mask(self, task_id):
        mask_path = self._storage_path / task_id / 'mask.png'
        stream = self._fopen(mask_path, 'rb')
        stream_len = os.path.getsize(mask_path)
        return stream, stream_len
