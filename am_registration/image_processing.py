import cv2
import numpy as np


def erode_dilate(image, kernel=5):
    img = image.copy()
    img = cv2.erode(img, np.ones((kernel, kernel), np.uint8), iterations=3)
    img = cv2.dilate(img, np.ones((kernel, kernel), np.uint8), iterations=1)
    return img


def find_am_centers(image):
    img = image.copy()
    contours, hierarchy = cv2.findContours(img.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    am_centers = []
    for c in contours:
        M = cv2.moments(c)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        am_centers.append((y, x))

    return np.array(am_centers, ndmin=2)
