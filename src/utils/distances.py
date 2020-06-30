import torch
import math
import numpy as np


def d2(x1, y1, x2, y2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    return torch.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def normal(x1, y1, x2, y2, sigma):
    return torch.exp(-(d2(x1=x1, y1=y1, x2=x2, y2=y2)/sigma)**2 * 0.5)

def d2_torch_points(p1, p2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    return torch.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

def get_closest_point_on_line(p, line):
    x_c = p[0]
    y_c = p[1]

    x_a = line[0][0]
    y_a = line[0][1]

    x_b = line[1][0]
    y_b = line[1][1]

    numerator = (x_a * x_b - x_c * x_b - x_a ** 2 + x_a * x_c - y_a ** 2 + y_c * y_a + y_a * y_b - y_c * y_b)
    denominator = (2 * x_a * x_b - x_b ** 2 - x_a ** 2 - y_a ** 2 + 2 * y_b * y_a - y_b ** 2)

    alpha = numerator / denominator

    x_result = x_a + alpha * (x_b - x_a)
    y_result = y_a + alpha * (y_b - y_a)

    return [x_result, y_result]

def get_closest_point_on_polygon(p, polygon):
    closest_point_on_polygon = [polygon[0][0], polygon[0][1]]
    shortest_dist_to_polygon = d2_torch_points(p, polygon[0])

    idx = 0

    for n in range(len(polygon) - 1):
        line = [polygon[n], polygon[n + 1]]

        closest_point_on_line = get_closest_point_on_line(p, line)
        shortest_dist_to_line = d2_torch_points(p, closest_point_on_line)

        if shortest_dist_to_line < shortest_dist_to_polygon:
            closest_point_on_polygon = closest_point_on_line
            shortest_dist_to_polygon = shortest_dist_to_line

            idx = n

    return closest_point_on_polygon, idx

def get_closest_point_and_angle_on_polygon(p, polygon):
    def _compute_angle(p1, p2):
        if torch.abs(p1[0] - p2[1]) < 0.001:
            if p1[1] > p2[1]:
                angle = torch.tensor(math.pi / 2.0)
            else:
                angle = torch.tensor(-math.pi / 2.0)
        else:
            if p1[0] < p2[1]:
                angle = torch.atan(
                    (p2[1] - p1[1]) / (p1[0] - p2[1]))
            else:
                if torch.abs(p1[1] - p2[1]) < 0.001:
                    angle = torch.tensor(math.pi / 2)
                else:
                    if p1[1] > p2[1]:
                        angle = torch.atan(
                            (p1[0] - p2[1]) / (
                                        p1[1] - p2[1])) + math.pi / 2
                    else:
                        angle = torch.atan(
                            (p1[0] - p2[1]) / (
                                        p1[1] - p2[1])) - math.pi / 2
        # ^ positive x direction is towards right and positive y direction is downwards.
        return angle

    closest_point_on_polygon, idx = get_closest_point_on_polygon(p, polygon)
    angle = _compute_angle(polygon[idx], polygon[idx+1])

    return closest_point_on_polygon, idx, angle

def get_gt_labels(polygon_pred, polygon_gt):
    result_polygon = [polygon_gt[0]]

    for point in polygon_pred[1:-1]:
        result_polygon.append(get_closest_point_on_polygon(point, polygon_gt))

    result_polygon.append(polygon_gt[-1])

    return result_polygon

def get_median_diff(start_points):
    diff = []
    N = len(start_points)

    if N > 0:
        for n in range(0, N - 1):
            x1 = start_points[n][0]
            y1 = start_points[n][1]
            x2 = start_points[n + 1][0]
            y2 = start_points[n + 1][1]
            diff.append(d2(x1, y1, x2, y2))

        return np.median(diff)
    else:
        return -1

def get_smallest_distance(point, point_list):
    d_min = 1e10

    for p in point_list:
        d = d2(p[0], p[1], point[0], point[1])
        if d > 0:
            d_min = min(d_min, d)

    return d_min
