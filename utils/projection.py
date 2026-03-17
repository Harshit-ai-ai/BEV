import cv2
import numpy as np

def get_bev(image):
    h, w = image.shape[:2]

    # Define source points (trapezoid on road)
    src = np.float32([
        [w * 0.4, h * 0.6],   # left top
        [w * 0.6, h * 0.6],   # right top
        [w * 0.1, h * 1.0],   # left bottom
        [w * 0.9, h * 1.0]    # right bottom
    ])

    # Destination points (rectangle)
    dst = np.float32([
        [0, 0],
        [256, 0],
        [0, 256],
        [256, 256]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(image, M, (256, 256))

    return bev