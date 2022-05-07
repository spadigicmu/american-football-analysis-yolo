import numpy as np
import cv2
from scipy import ndimage
import math

from transform import four_point_transform


def get_image_with_lines(image1):
    """Find the most vertical line and rotate the image by that angle"""
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    canimg = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, 140, np.array([]))
    # print(lines)
    # lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
    angle_to_rotate = 0
    # print("line", lines)

    for line in lines:
        # tmp_image = image1.copy()
        rho, theta = line[0]
        # print(str(rho) + " : " + str(get_angle_degree(theta)))

        # angle_to_rotate = theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # print("theta in degrees: ", (theta * 180) / 3.1415926)
        if theta < 0.8:
            angle_to_rotate = theta
        elif theta > (math.pi - 0.8):
            angle_to_rotate = math.pi + theta
        # if get_angle_degree(theta) == 0.0:
        cv2.line(image1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.putText(tmp_image, str(rho), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             2)
        # cv2.imshow("Image", tmp_image)
        # cv2.waitKey(0)

    return image1, angle_to_rotate, lines


def get_most_vertical_line(lines):
    min_target = 0
    max_target = math.pi
    val = None
    max_val = None
    for line in lines:
        rho, theta = line[0]
        if val is None:
            val = theta
        elif min(theta - min_target, max_target - theta) < val:
            val = theta

    return val


def get_vertical_extremes(h, w, big_h, big_w, angle_to_rotate):
    bottom_point = (100 + h * math.sin(angle_to_rotate), big_h - 100)
    top_point = (100 + w * math.cos(angle_to_rotate), 100)
    return (top_point, bottom_point)


def get_rho(point, phi):
    """
    x, y = point
    m = math.tan(theta)
    return abs(y - m * x) / math.sqrt(m * m + 1)
    """
    x, y = point
    theta = phi - math.pi/2

    m = math.tan(theta)

    rho = (y - m*x) / (math.sin(phi) - m*math.cos(phi))

    return rho


def get_coordinates(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    return x1, y1, x2, y2


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)


def get_angle_degree(theta):
    return (theta * 180) / math.pi


def get_image_with_max_lines(image1):
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    canimg = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, 140, np.array([]))

    print(lines)

    # print(lines)
    # lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
    angle_to_rotate = 0
    # print("line", lines)
    # print("printing lines")

    degree_lines = []
    left_most_line = None
    right_most_line = None

    min_rho = None
    max_rho = None

    deg_lines = []
    for line in lines:
        tmp_image = image1.copy()
        rho, theta = line[0]
        deg_lines.append((rho, theta * 180 / 3.1415926))

        x1, y1, x2, y2 = get_coordinates(rho, theta)
        cv2.line(tmp_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(tmp_image, str(rho), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow("Image", tmp_image)
        # cv2.waitKey(0)

    # print(deg_lines)

    for line in lines:
        rho, theta = line[0]
        degree_theta = get_angle_degree(theta)
        # print(rho + " : " + get_angle_degree(theta))

        # Only include lines that are more vertical than horizontal
        if (135 <= degree_theta <= 180) or (0 <= degree_theta <= 45):
            # degree_lines.append((rho, theta))

            if min_rho is None:
                min_rho = rho
                right_most_line = (min_rho, theta)
            elif rho < min_rho:
                min_rho = rho
                right_most_line = (min_rho, theta)

            if max_rho is None:
                max_rho = rho
                left_most_line = (max_rho, theta)
            elif rho > max_rho:
                max_rho = rho
                left_most_line = (max_rho, theta)

    degree_lines.append(left_most_line)
    # degree_lines.append((left_most_line[0] + 20, left_most_line[1] + math.pi/2))
    degree_lines.append(right_most_line)
    # degree_lines.append((right_most_line[0] + 240, right_most_line[1] + math.pi/2))
    # degree_lines.append(
    #     (right_most_line[0] - 260, right_most_line[1] + math.pi / 2))

    # print(degree_lines)

    for filt_deg in degree_lines:
        x1, y1, x2, y2 = get_coordinates(filt_deg[0], filt_deg[1])
        cv2.line(image1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image1, angle_to_rotate, degree_lines


def plot_image_with_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = get_coordinates(line[0], line[1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, str(line[0]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    return img


# image = cv2.imread("img1.jpg", cv2.IMREAD_UNCHANGED)
image = cv2.imread("img1.jpg", cv2.IMREAD_UNCHANGED)
h, w, c = image.shape
image1 = image.copy()
image2 = image.copy()

image_with_lines, angle_to_rotate, lines = get_image_with_lines(image)
cv2.imwrite("lines.png", image_with_lines)

most_vertical_angle = get_most_vertical_line(lines)

# print("angle to rotate", angle_to_rotate)
# print(rho)
# Rotate image by specified angle
img_rotated = ndimage.rotate(image2, 180 * angle_to_rotate / math.pi)

img_rotated = cv2.copyMakeBorder(img_rotated, 100, 100, 100, 100,
                                 cv2.BORDER_CONSTANT)
cv2.imwrite("rotated.png", img_rotated)

img_rotated_og = img_rotated.copy()

final_h, final_w, final_c = img_rotated.shape
top_most_point, bottom_most_point = get_vertical_extremes(w, h, final_w,
                                                          final_h,
                                                          angle_to_rotate)

# perpendicular_theta = angle_to_rotate + most_vertical_angle + math.pi / 2

perpendicular_theta = most_vertical_angle - angle_to_rotate + math.pi / 2

rho_t = get_rho(top_most_point, perpendicular_theta)
rho_b = get_rho(bottom_most_point, perpendicular_theta)
# rho_t = 200
# rho_b = 500

top_line = (rho_t, perpendicular_theta)
bottom_line = (rho_b, perpendicular_theta)
# cv2.imwrite("edges.png", edges)

# Filter lines to get only the left most and right most lines
rotated_image_with_lines, angle, vertical_lines = get_image_with_max_lines(
    img_rotated)
cv2.imwrite("rotated_lines.png", rotated_image_with_lines)

left_line = vertical_lines[0]
right_line = vertical_lines[1]

img_rotated_borders = img_rotated.copy()

img_rotated_borders = plot_image_with_lines(img_rotated_borders, [left_line,
                                                                  right_line,
                                                                  top_line,
                                                                  bottom_line])

cv2.imwrite("four_borders.png", img_rotated_borders)


x1, y1 = intersection(left_line, top_line)
x2, y2 = intersection(right_line, top_line)
x3, y3 = intersection(right_line, bottom_line)
x4, y4 = intersection(left_line, bottom_line)

img_rotated_tmp = img_rotated.copy()

cv2.line(img_rotated_tmp, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.line(img_rotated_tmp, (x2, y2), (x3, y3), (0, 0, 255), 2)
cv2.line(img_rotated_tmp, (x3, y3), (x4, y4), (0, 0, 255), 2)
cv2.line(img_rotated_tmp, (x4, y4), (x1, y1), (0, 0, 255), 2)

img_rotated_tmp = img_rotated.copy()

cv2.imwrite("trapezium.png", img_rotated_tmp)

pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

# print(pts)

warped = four_point_transform(img_rotated_og, pts)

# print(angle)

cv2.imwrite("warped_image.png", warped)



# first get the most vertical line in original image

# after rotating get the height of the image to store the top and bottom of the rotated image

# draw the perpendicular lines to the side lines and get the trapezium coords
