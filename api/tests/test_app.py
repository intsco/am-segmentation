from unittest.mock import MagicMock
import falcon
from falcon import testing
import pytest
import json

from api.app import create_app


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def client(mock_store):
    api = create_app(mock_store)
    return testing.TestClient(api)


# pytest will inject the object returned by the "client" function
# as an additional parameter.
def test_list_images(client):
    doc = {
        'images': [
            {
                'href': '/images/1eaf6ef1-7f2d-4ecc-a8d5-6e8adba7cc0e.png'
            }
        ]
    }

    response = client.simulate_get('/images')
    result_doc = json.loads(response.content, encoding='utf-8')

    assert result_doc == doc
    assert response.status == falcon.HTTP_OK


# With clever composition of fixtures, we can observe what happens with
# the mock injected into the image resource.
def test_post_image(client, mock_store):
    file_name = 'fake-image-name.xyz'
    # We need to know what ImageStore method will be used
    mock_store.save.return_value = file_name
    image_content_type = 'image/xyz'

    response = client.simulate_post(
        '/images',
        body=b'some-fake-bytes',
        headers={'content-type': image_content_type}
    )

    assert response.status == falcon.HTTP_CREATED
    assert response.headers['location'] == '/images/{}'.format(file_name)
    saver_call = mock_store.save.call_args

    # saver_call is a unittest.mock.call tuple. It's first element is a
    # tuple of positional arguments supplied when calling the mock.
    assert isinstance(saver_call[0][0], falcon.request_helpers.BoundedStream)
    assert saver_call[0][1] == image_content_type
