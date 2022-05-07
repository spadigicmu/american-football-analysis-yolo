import json
import math

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy import ndimage


def get_bbox_list():
    # Opening JSON file
    f = open('bbox.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # plot the bbox with image
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    img = cv2.imread("../input_images/img1.jpg", cv2.IMREAD_UNCHANGED)
    ax.imshow(img)
    bbox_list = []
    for box in data['predictions']:
        cur_box = box
        curr_bbox = []
        curr_bbox.append(cur_box['x'])
        curr_bbox.append(cur_box['y'])
        bbox_list.append(curr_bbox)
        # Create a Rectangle patch
        rect2 = patches.Rectangle((cur_box['x'] - (cur_box['width'] / 2),
                                   cur_box['y'] - (cur_box['height'] / 2)),
                                  cur_box['width'], cur_box['height'],
                                  linewidth=1, edgecolor='b', facecolor='none')
        # Add the patch to the Axes

        ax.add_patch(rect2)
    plt.show()
    bbox_list = np.array(bbox_list)
    # print(bbox_list)
    return bbox_list


def plot_points(img, points):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        cv2.imshow("img_circle", img)
        cv2.waitKey(0)
    cv2.imwrite("../output_images/circles_img.png", img)


def get_rotation_matrix(angle_to_rotate):
    n = np.array([[math.cos(angle_to_rotate), math.sin(angle_to_rotate)],
                  [-math.sin(angle_to_rotate), math.cos(angle_to_rotate)]])
    # n = np.array([[math.cos(angle_to_rotate), -math.sin(angle_to_rotate)],
    #               [math.sin(angle_to_rotate), math.cos(angle_to_rotate)]])

    return n


def rotate_operation(points, angle_to_rotate, h, w):
    rotation_matrix = get_rotation_matrix(angle_to_rotate)
    # Get the size of the image after rotation (with padding)
    big_img_w = int(abs(w * math.cos(angle_to_rotate)) + abs(h * math.sin(angle_to_rotate)))
    big_img_h = int(abs(w * math.sin(angle_to_rotate)) + abs(h * math.cos(angle_to_rotate)))

    # The rotation on image is done using scipy.ndimage.rotate which pivots the image at the center
    # But doing the rotation using a rotation matrix pivots it at the top left corner (old_origin).
    # 1. We compute the translation of the old origin with respect to the image center
    # 2. We rotate the image using the top left corner as pivot
    # 3. Now the new positions of all points need to be translated by the amount the old_origin moved
    old_origin = np.array([-big_img_w / 2, -big_img_h / 2])
    new_origin = apply_matrix_operation(rotation_matrix, [old_origin])
    movement = new_origin[0] - old_origin

    rotated_points = apply_matrix_operation(rotation_matrix, points)
    r_t_points = translate_operation(rotated_points, movement[0], movement[1])
    return r_t_points


def translate_operation(points, x_t, y_t):
    for point in points:
        point[0] += x_t
        point[1] += y_t

    return points


def apply_matrix_operation(matrix, points):
    transformed_points = []
    for point in points:
        transformed_points.append(np.dot(matrix, point))

    return transformed_points


def get_rotated_reference(img, angle_to_rotate):
    img_rotated = ndimage.rotate(img, 180 * angle_to_rotate / math.pi)

    # img_rotated = cv2.copyMakeBorder(img_rotated, 100, 100, 100, 100,
    #                                  cv2.BORDER_CONSTANT)
    # cv2.imwrite("rotated_sample.png", img_rotated)

    return img_rotated


angle_to_rotate = math.pi / 4
img = cv2.imread("../input_images/img1.jpg", cv2.IMREAD_UNCHANGED)
h, w, c = img.shape
points = get_bbox_list()
rotated_points = rotate_operation(points, angle_to_rotate, h, w)
# translated_points = translate_operation(rotated_points, 100, 100)

# rotated_img = cv2.imread("rotated_sample.png", cv2.IMREAD_UNCHANGED)
rotated_img = get_rotated_reference(img, 0)
plot_points(rotated_img, rotated_points)
