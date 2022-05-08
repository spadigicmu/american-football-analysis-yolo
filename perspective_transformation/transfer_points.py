# input bounding box centers, warped image

# get lines in the image, at this point all the lines should be vertical

# get the distance between a pair of lines in the middle -> d1

# store the distance between a pair of lines in the bird's eye view template
# -> d2

# use d1 and d2 to compute the ratio for transforming the x axis of the
# points onto the template

# based on which line you want to use as reference, add corresponding value to
# the x axis co-ord, that will shift the players across the required lines

# similarly get the ratio for the y axis bet getting height of warped image
# and template

# now we have the transformed coordinates required for the plotting on template
import copy

import cv2
import json
import math
import numpy as np


class TransferPoints:
    def __init__(self, birds_eye_template_path):
        self.template_image = cv2.imread(birds_eye_template_path,
                                         cv2.IMREAD_UNCHANGED)

        self.template_image_height, self.template_image_width = \
            self.fetch_image_dim(self.template_image)

        self.template_lines = self.fetch_image_lines(self.template_image,
                                                     vote_threshold=350)

        self.template_vertical_gap = self.fetch_avg_vertical_line_dist(
            self.template_lines, self.template_image_width)
        self.template_horizontal_gap = self.fetch_avg_horizontal_line_dist(
            self.template_lines, self.template_image_height)

    def fetch_image_dim(self, image):
        dimensions = image.shape
        height = dimensions[0]
        width = dimensions[1]
        # print(dimensions)
        return height, width

    def get_coordinates(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        return x1, y1, x2, y2

    def get_angle_degree(self, theta):
        return (theta * 180) / math.pi

    def fetch_image_lines(self, image, vote_threshold=140):
        """Find the most vertical line and rotate the image by that angle"""
        image1 = image.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        canimg = cv2.Canny(gray, 50, 200)
        lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, vote_threshold,
                               np.array([]))
        # print(lines)
        # lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
        angle_to_rotate = 0
        # print("line", lines)
        cartesian_lines = []
        for line in lines:
            # tmp_image = image1.copy()
            rho, theta = line[0]
            # print(str(rho) + " : " + str(get_angle_degree(theta)))

            """
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
            """

            x1, y1, x2, y2 = self.get_coordinates(rho, theta)

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
            cartesian_lines.append((x1, y1, x2, y2))

        # cv2.imshow("Image with lines", image1)
        # cv2.waitKey(0)
        # return image1, angle_to_rotate, lines
        return lines

    def filter_vertical_lines(self, lines):
        vertical_lines = []

        for line in lines:
            rho, theta = line[0]
            degree_theta = self.get_angle_degree(theta)
            # print("degree theta(theta) ", degree_theta, theta, "rho: ", rho)

            # Only include lines that are more vertical than horizontal
            if (170 <= degree_theta <= 180) or (0 <= degree_theta <= 10):
                vertical_lines.append(line)

        return vertical_lines

    def filter_horizontal_lines(self, lines):
        horizontal_lines = []

        for line in lines:
            rho, theta = line[0]
            degree_theta = self.get_angle_degree(theta)
            # print("degree theta(theta) ", degree_theta, theta, "rho: ", rho)

            # Only include lines that are more vertical than horizontal
            if (80 <= degree_theta <= 100):  # or (0 <= degree_theta <= 10):
                horizontal_lines.append(line)

        return horizontal_lines

    def similar_line_exists(self, new_val, existing_val_arr, range_lim):

        for existing_val in existing_val_arr:
            if abs(existing_val - new_val) <= range_lim:
                return True

        return False

    def get_avg_diff(self, val_arr):
        differences = []

        for i in range(1, len(val_arr)):
            differences.append(val_arr[i] - val_arr[i - 1])

        avg_diff = sum(differences) / len(differences)
        return avg_diff

    def fetch_avg_vertical_line_dist(self, lines, img_width):
        vertical_lines = self.filter_vertical_lines(lines)

        diff_x = [0, img_width]

        for vertical_line in vertical_lines:
            rho, theta = vertical_line[0]

            if not self.similar_line_exists(rho, diff_x, 5):
                diff_x.append(rho)

        diff_x = sorted(diff_x)

        avg_gap = self.get_avg_diff(diff_x)

        return avg_gap
        # pass

    def fetch_avg_horizontal_line_dist(self, lines, img_height):

        horizontal_lines = self.filter_horizontal_lines(lines)

        diff_x = []

        for horizontal_line in horizontal_lines:
            rho, theta = horizontal_line[0]

            if not self.similar_line_exists(rho, diff_x, 5):
                diff_x.append(rho)

        diff_x = sorted(diff_x)

        avg_gap = self.get_avg_diff(diff_x)

        return avg_gap

        # return img_height

    def fetch_tranfer_point_coord(self, bbox_coords, height_ratio,
                                  width_ratio, x_offset):

        current_points = copy.deepcopy(bbox_coords)
        for entry in range(len(current_points['predictions'])):
            current_points['predictions'][entry]['x'] = x_offset + current_points['predictions'][entry]['x'] * width_ratio
            current_points['predictions'][entry]['y'] = current_points['predictions'][entry]['y'] * height_ratio

        return current_points

    def plot_points(self, input_img, points):
        img = input_img.copy()
        for point in points['predictions']:
            cv2.circle(img, (int(point['x']), int(point['y'])), 5, (255, 0, 0), -1)
            # cv2.imshow("img_circle", img)
            # cv2.waitKey(0)

        return img

    def transfer_points(self, input_img_path, bbox_file):

        input_image = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
        input_image_height, input_image_width = \
            self.fetch_image_dim(input_image)

        input_image_lines = self.fetch_image_lines(input_image,
                                                   vote_threshold=140)

        input_image_vertical_gap = self.fetch_avg_vertical_line_dist(
            input_image_lines, input_image_width)
        input_image_horizontal_gap = self.fetch_avg_horizontal_line_dist(
            input_image_lines, input_image_height)

        height_ratio = self.template_horizontal_gap / input_image_horizontal_gap
        width_ratio = self.template_vertical_gap / input_image_vertical_gap

        bbox_coords = json.load(open(bbox_file))

        transfer_coords = self.fetch_tranfer_point_coord(bbox_coords, height_ratio, width_ratio, self.template_vertical_gap*2)

        birds_eye_img = self.plot_points(self.template_image, bbox_coords)
        cv2.imwrite("output_images/birds_eye_img.png", birds_eye_img)
        print(input_image_vertical_gap)


if __name__ == "__main__":
    transfer_points = TransferPoints('input_images/birds_eye_view_field.png')
    transfer_points.transfer_points('output_images/warped_image.png',
                                    'perspective_transformation/'
                                    'rotated_bbox.json')

    # transfer_points.fetch_image_dim()
