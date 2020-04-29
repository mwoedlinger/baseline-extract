import numpy as np

def d2_np(u, v):
    return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def is_in_box(point, left, top, width, height):
    x = point[0]
    y = point[1]

    if left < x < left + width:
        if top < y < top + height:
            return True

    return False


def contains_start_point(points, bl_stats):
    comp_contains_sp = []
    for n, bl_s in enumerate(bl_stats):

        left = bl_s[0]
        top = bl_s[1]
        width = bl_s[2]
        height = bl_s[3]

        if width > height:
            left -= 5
            width += 10
        else:
            top -= 5
            height += 10

        contains_sp_switch = False
        for p in points:
            if is_in_box(p, left, top, width, height):
                comp_contains_sp.append(True)
                contains_sp_switch = True
                break

        if not contains_sp_switch:
            comp_contains_sp.append(False)

    return comp_contains_sp


def get_cc_array(stats, area_thresh=100):
    cc_array = np.zeros((len(stats), 4, 2))

    for n, cc in enumerate(stats):
        if cc[2] * cc[3] < area_thresh:
            top_left = -10000
            top_right = -10000
            bottom_left = -10000
            bottom_right = -10000
        else:
            top_left = np.array([cc[0], cc[1]])
            top_right = np.array([cc[0] + cc[2], cc[1]])
            bottom_left = np.array([cc[0], cc[1] + cc[3]])
            bottom_right = np.array([cc[0] + cc[2], cc[1] + cc[3]])

        cc_array[n, 0, :] = top_left
        cc_array[n, 1, :] = top_right
        cc_array[n, 2, :] = bottom_left
        cc_array[n, 3, :] = bottom_right

    return cc_array


def get_angles(cc_array, start_points):
    dist_array = -np.ones(cc_array.shape[0:2])  # The minus is for debug
    angles = np.zeros(len(start_points))

    for sp_idx, p in enumerate(start_points):
        for n in range(cc_array.shape[0]):
            for k in range(0, 4):
                dist_array[n, k] = d2_np(p, cc_array[n, k, :])

        cc_idx, p_idx = np.unravel_index(np.argmin(dist_array, axis=None), dist_array.shape)

        if np.min(dist_array) < 0:
            print('something went wrong!')
            return 0

        bb_w = (cc_array[cc_idx, 1, 0] - cc_array[cc_idx, 0, 0])
        bb_h = (cc_array[cc_idx, 2, 1] - cc_array[cc_idx, 0, 1])

        if p_idx == 0:  # top_left
            angle = np.arctan(-bb_h / bb_w)
        elif p_idx == 1:  # top_right
            angle = np.arctan(-bb_w / bb_h) - np.pi / 2.0
        elif p_idx == 2:  # bottom_left
            angle = np.arctan(bb_h / bb_w)
        elif p_idx == 3:  # bottom_right
            angle = np.arctan(bb_w / bb_h) + np.pi / 2.0
        else:
            print('p_idx = {}'.format(p_idx))

        if np.pi * 0.75 < abs(angle) < np.pi * 1.25:
            #print('angle {} reset to 0 for point {}'.format(angle, sp_idx))
            angle = 0

        angles[sp_idx] = angle

    return angles