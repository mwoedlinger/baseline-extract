import torch


def d2(x1, y1, x2, y2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    return torch.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def point_line_distance(x_0, y_0, xp_1, yp_1, xp_2, yp_2):
    # Taken from: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return torch.abs((yp_2-yp_1)*x_0 - (xp_2-xp_1)*y_0 + xp_2*yp_1 - yp_2*xp_1)/torch.sqrt((yp_2-yp_1)**2 + (xp_2-xp_1)**2)
