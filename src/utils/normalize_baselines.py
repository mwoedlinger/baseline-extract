import torch


def d2(x1, y1, x2, y2):
    """
    Computes the euclidian distance between the two points (x1, y1) and (x2, y2).
    """
    return torch.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def normalize_baselines(bl: list, segment_length: float, device: str):
    """
    Takes a baseline as input and returns a baseline that consists of normalized line segments.
    That means that the baseline consists of a list of points where each point is exactly 'segment_length' away
    from the next point.
    :param baseline: The input baseline
    :param segment_length: The distance between two points.
    :return: a baseline
    """

    new_bl = [bl[0]]
    # new_bl = bl[0].unsqueeze(0).to(device)

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
                # p = torch.tensor([[x,y]]).to(device)
                # new_bl = torch.cat([new_bl, p], dim=0)
                idx = i
                break
            else:
                if i == len(bl)-1:
                    new_bl.append(bl[-1]) #TODO: think of something better
                    # new_bl = torch.cat([new_bl, bl[-1].unsqueeze(0).to(device)], dim=0)
                    done = True
        if done:
            break

    return torch.cat([l.unsqueeze(0) for l in new_bl], dim=0)
    # return new_bl