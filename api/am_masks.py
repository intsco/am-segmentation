import re
import falcon
import json

ALLOWED_IMAGE_TYPES = (
    'image/png',
)
_UUID_PATTERN = re.compile(
    '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)


def validate_image_type(req, resp, resource, params):
    if req.content_type not in ALLOWED_IMAGE_TYPES:
        msg = f'Image type not allowed. Must be one of {ALLOWED_IMAGE_TYPES}'
        raise falcon.HTTPBadRequest('Bad request', msg)


def validate_task_id(req, resp, resource, params):
    # Always validate untrusted input!
    if not _UUID_PATTERN.match(params.get('task_id', '')):
        raise IOError('Wrong task id')


# class AblationMaskCollection(object):
#
#     def __init__(self, segmentator):
#         self._segmentator = segmentator
#
#     def on_get(self, req, resp):
#         mask_docs = [{'href': f'/masks/{fn}'}
#                       for fn in self._image_store.list_masks()]
#         doc = {
#             'masks': mask_docs
#         }
#         resp.body = json.dumps(doc, ensure_ascii=False)
#         resp.status = falcon.HTTP_200


class AblationMask(object):

    def __init__(self, segmentator):
        self._segmentator = segmentator

    @falcon.before(validate_task_id)
    def on_get(self, req, resp, task_id):
        try:
            resp.stream, resp.stream_len = self._segmentator.read_mask(task_id)
            resp.content_type = falcon.MEDIA_PNG
        except IOError as e:
            raise falcon.HTTPNotFound()


class SegmentationTaskCollection(object):

    def __init__(self, segmentator):
        self._segmentator = segmentator

    @falcon.before(validate_image_type)
    def on_post(self, req, resp):
        task_id = self._segmentator.create_task(req.stream, req.content_type)
        resp.status = falcon.HTTP_201
        resp.location = '/tasks/' + task_id


class SegmentationTask(object):

    def __init__(self, segmentator):
        self._segmentator = segmentator

    @falcon.before(validate_task_id)
    def on_get(self, req, resp, task_id):
        doc = {
            'status': self._segmentator.task_status(task_id)
        }
        resp.body = json.dumps(doc, ensure_ascii=False)
        resp.status = falcon.HTTP_200
