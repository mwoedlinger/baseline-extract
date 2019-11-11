import torch
import numpy as np


def d2(x1, y1, x2, y2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    return torch.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def normal(x1, y1, x2, y2, sigma):
    return torch.exp(-(d2(x1=x1, y1=y1, x2=x2, y2=y2)/sigma)**2 * 0.5)

def point_line_distance(x_0, y_0, xp_1, yp_1, xp_2, yp_2):
    # Taken from: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return torch.abs((yp_2-yp_1)*x_0 - (xp_2-xp_1)*y_0 + xp_2*yp_1 - yp_2*xp_1)/torch.sqrt((yp_2-yp_1)**2 + (xp_2-xp_1)**2)


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