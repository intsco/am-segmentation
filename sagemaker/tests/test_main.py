import numpy as np
import cv2
from matplotlib import pyplot as plt
from sagemaker.main import model_fn, input_fn, predict_fn, output_fn


if __name__ == '__main__':
    model = model_fn('.')

    with open('tests/013.png', 'rb') as f:
        input_data = f.read()

    inputs = input_fn(input_data, 'application/x-image')
    masks = predict_fn(inputs, model)
    output_data = output_fn(masks, 'application/x-image')

    with open('/tmp/1.png', 'wb') as f:
        f.write(output_data)

    buf = np.frombuffer(output_data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    plt.imshow(img)
    plt.show()
