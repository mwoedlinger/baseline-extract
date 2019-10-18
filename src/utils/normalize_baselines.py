import numpy as np


def d2(x1, y1, x2, y2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def normalize_baselines(bl: list, segment_length: float):
    """
    Takes a baseline as input and returns a baseline that consists of normalized line segments.
    That means that the baseline consists of a list of points where each point is exactly 'segment_length' away
    from the next point.
    :param baseline: The input baseline
    :param segment_length: The distance between two points.
    :return: a baseline
    """

    new_bl = [bl[0]]
    idx = 1  # Index of next baseline point of original label
    x, y = bl[0]
    done = False


    for _ in range(1000):
        for i in range(idx, len(bl)):
            x_n, y_n = bl[i]
            dist_to_next = d2(x, y, x_n, y_n)

            if dist_to_next > segment_length:
                dist_ratio = segment_length / dist_to_next

                x = (1 - dist_ratio) * x + dist_ratio * x_n
                y = (1 - dist_ratio) * y + dist_ratio * y_n

                new_bl.append([x, y])
                idx = i
                break
            else:
                if i == len(bl)-1:
                    new_bl.append(bl[-1]) #TODO: think of something better
                    done = True
        if done:
            break
    return new_bl