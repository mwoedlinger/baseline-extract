import torch
import math
import random
from .distances import d2

def normalize_baselines(bl_par: list, segment_length: float):
    """
    Takes a baseline as input and returns a baseline that consists of normalized line segments.
    That means that the baseline consists of a list of points where each point is exactly 'segment_length' away
    from the next point.
    :param baseline: The input baseline
    :param segment_length: The distance between two points.
    :return: a baseline
    """

    bl = bl_par
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

                new_bl.append(torch.tensor([x, y]))
                idx = i
                break
            else:
                if i == len(bl)-1:
                    new_bl.append(bl[-1]) #TODO: think of something better
                    done = True
        if done:
            break

    return torch.cat([l.unsqueeze(0) for l in new_bl], dim=0)

def compute_start_and_angle(baseline, idx, data_augmentation=False, box_size=None):
    """
    For a given baseline returns the point at index=idx and the angle the baseline segment [idx, idx+1] has
    towards a horizontal line.
    NOTE: The angle gets a negative sign compared to the mathematical positive direction because
    the positive y direction is downwards.
    :param baseline: The baseline as a torch.tensor
    :param idx: The index
    :param data_augmentation: if set to True perturbs the point and angle randomly by small values
    :return: a tuple of x coordinate of the point, y coordinate and the angle
    """

    if torch.abs(baseline[idx, 0] - baseline[idx + 1, 0]) < 0.001:
        if baseline[idx, 1] > baseline[idx + 1, 1]:
            angle = torch.tensor(math.pi / 2.0)
        else:
            angle = torch.tensor(-math.pi / 2.0)
    else:
        if baseline[idx, 0] < baseline[idx + 1, 0]:
            angle = torch.atan((baseline[idx + 1, 1] - baseline[idx, 1]) / (baseline[idx, 0] - baseline[idx + 1, 0]))
        else:
            if torch.abs(baseline[idx, 1] - baseline[idx + 1, 1]) < 0.001:
                angle = torch.tensor(math.pi / 2)
            else:
                if baseline[idx, 1] > baseline[idx + 1, 1]:
                    angle = torch.atan(
                        (baseline[idx, 0] - baseline[idx + 1, 0]) / (baseline[idx, 1] - baseline[idx + 1, 1])) + math.pi/2
                else:
                    angle = torch.atan(
                        (baseline[idx, 0] - baseline[idx + 1, 0]) / (baseline[idx, 1] - baseline[idx + 1, 1])) - math.pi/2
    # ^ positive x direction is towards right and positive y direction is downwards.

    x = baseline[idx, 0].clone()
    y = baseline[idx, 1].clone()

    # Perturb position and angle, otherwise the network will learn to always predict angle = 0
    if data_augmentation:
        if box_size is not None:
            x += random.randint(-box_size/4, box_size/4)
            y += random.randint(-box_size/4, box_size/4)
        else:
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)
        angle += random.uniform(-0.3, 0.3)

    return x, y, angle