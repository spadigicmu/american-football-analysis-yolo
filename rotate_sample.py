import numpy as np
import cv2
from scipy import ndimage
import math

angle_to_rotate = math.pi/2

img = cv2.imread("img1.jpg", cv2.IMREAD_UNCHANGED)

img_rotated = ndimage.rotate(img, 180 * angle_to_rotate / math.pi)

img_rotated = cv2.copyMakeBorder(img_rotated, 100, 100, 100, 100,
                                 cv2.BORDER_CONSTANT)
cv2.imwrite("rotated_sample.png", img_rotated)

