import numpy as np
import cv2
import torch
import torch.nn as nn
import math
from torchvision import models
from ..segmentation.gcn_model import GCN
from ..utils.distances import get_smallest_distance, get_median_diff
from ..utils.utils import load_class_dict


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


class LineDetector:
    """
    Loads the line rider model for inference.
    """

    def __init__(self, config: dict):
        self.config = config
        line_rider_weights = config['line_rider']['weights']
        line_finder_weights = config['line_finder']['weights']
        self.device_lr = torch.device(
            'cuda:' + str(config['line_rider']['device']) if torch.cuda.is_available() else 'cpu')
        self.device_lf = torch.device(
            'cuda:' + str(config['line_finder']['device']) if torch.cuda.is_available() else 'cpu')
        self.classes, self.colors, self.class_dict = load_class_dict(config['line_finder']['class_file'])
        self.model = config['line_finder']['model']
        self.backbone = config['line_finder']['backbone']
        self.num_classes = len(self.classes)
        self.class_idx = {self.classes[idx]: idx for idx in range(0, self.num_classes)}
        self.backbone = config['line_finder']['backbone']
        self.auto_generate_start_points = config['line_finder']['auto_generate_start_points']

        print('## Load Line Rider:')
        self.line_rider = self.load_line_rider_model(line_rider_weights, self.device_lr)
        if True:#config['line_rider']['with_segmentation']:
            self.classes, _, _ = load_class_dict(config['line_finder']['class_file'])
            print('## Load Line Finder:')
            self.line_finder_seg = self.load_line_finder_model(line_finder_weights, self.device_lf, )
            print('## Loaded!')

    def load_line_rider_model(self, weights, device):
        model = torch.load(weights, map_location=device)
        model.to(device)
        model.eval()
        model.device = device

        return model

    def load_line_finder_model(self, weights, device):
        if self.model == 'GCN':
            seg_model = GCN(n_classes=self.num_classes, resnet_depth=self.backbone)
        elif self.model == 'DeepLabV3':
            seg_model = models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes)
        seg_model.load_state_dict(torch.load(weights, map_location=device))
        seg_model.to(device)
        seg_model.eval()

        return seg_model

    @staticmethod
    def segmentation_postprocessing(array: np.array, sigma, threshold, open_kernel):
        """
        Applies post processing steps to the segmentation output to better extract the start points.
        """
        seg_blur = cv2.GaussianBlur(array, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        seg_thresh = (seg_blur > threshold) * 1.0
        seg_open = cv2.morphologyEx(seg_thresh, cv2.MORPH_OPEN, (open_kernel, open_kernel))
        out = seg_open

        return out

    def extract_start_points_and_angles(self, seg: np.array) -> tuple:
        """
        Extract start points and angles from the segmantation output.
        :param seg: Segmentation output as a numpy array
        :return: The tuple: sp_labels, ep_labels, angles, label_list
        """
        segmentation = np.transpose(seg[0], (1, 2, 0))

        probs_start_points = segmentation[:, :, self.class_idx['start_points']]  # TODO: carefull
        probs_end_points = segmentation[:, :, self.class_idx['end_points']]  # TODO: carefull
        probs_baselines = segmentation[:, :, self.class_idx['baselines']]
        probs_border = segmentation[:, :, self.class_idx['text']]

        # Apply postprocessing:
        # Postprozessing parameters
        sigma = 0.3
        threshold = 0.5#TODO: change threshold!!!
        threshold_sp = 0.3
        open_kernel = 3

        # Extract baselines
        bl = self.segmentation_postprocessing(probs_baselines, sigma, threshold, open_kernel)
        _, bl_labels, bl_stats, bl_centroids = cv2.connectedComponentsWithStats(bl.astype(np.uint8))

        # Extract baseline_borders
        border = self.segmentation_postprocessing(probs_border, sigma, threshold, open_kernel)
        _, border_labels, _, _ = cv2.connectedComponentsWithStats(border.astype(np.uint8))

        # Extract start points
        sp = self.segmentation_postprocessing(probs_start_points, sigma, threshold_sp, open_kernel)
        _, _, _, start_points = cv2.connectedComponentsWithStats(sp.astype(np.uint8))

        # Extract end points
        ep = self.segmentation_postprocessing(probs_end_points, sigma, threshold, open_kernel)
        _, _, _, end_points = cv2.connectedComponentsWithStats(ep.astype(np.uint8))

        # Compute the angles and match start and end points
        probs_sum = probs_start_points + probs_end_points + probs_baselines + 0.5 * probs_border  # ?
        ps = cv2.GaussianBlur(probs_sum, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        ps = (ps > threshold) * 1.0
        ps = cv2.morphologyEx(ps, cv2.MORPH_OPEN, (3, 3))
        # TODO: remove start points that don't connect to a baseline
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ps.astype(np.uint8))

        # stats is a matrix where for every label l the vector stats[l] is given by:
        # [leftmost (x) coordinate, topmost (y) coordinate, width of bounding box, height of bouinding box, area]
        # see https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python

        # Assign labels to the start and end points
        # [1:] because the background is also a component


        ###########################
        label_list_tmp = []
        sp_list = []

        for sp in start_points[1:]:
            sp_list.append(sp)
            label_list_tmp.append(labels[(int(sp[1]), int(sp[0]))])

        cc_array = get_cc_array(stats)
        angles_list = get_angles(cc_array, sp_list)
        ep_labels_tmp = {labels[(int(p[1]), int(p[0]))]: p for p in end_points[1:]}


        sp_labels = {}
        ep_labels = {}
        angles = {}
        label_list = []

        for n in range(len(sp_list)):
            label_list.append(n)
            sp_labels.update({n: sp_list[n]})
            angles.update({n: angles_list[n]})
            if label_list_tmp[n] in ep_labels_tmp.keys():
                ep_labels.update({n: ep_labels_tmp[label_list_tmp[n]]})
        ###########################




        ###########################
        # sp_labels = {labels[(int(p[1]), int(p[0]))]: p for p in start_points[1:]}
        # ep_labels = {labels[(int(p[1]), int(p[0]))]: p for p in end_points[1:]}
        #
        # label_list = sp_labels.keys()
        # cc_array = get_cc_array(stats)
        # angles_list = get_angles(cc_array, start_points[1:])
        #
        # angles = {labels[(int(start_points[n + 1][1]), int(start_points[n + 1][0]))]: angles_list[n]
        #           for n in range(len(angles_list))}
        ###########################





        # angles = {l: np.arctan(stats[l][3] / stats[l][2]) for l in label_list}

        bl_out_truth = contains_start_point(start_points, bl_stats)

        #####################################################################################
        if self.auto_generate_start_points:
            # Generate new start points where the prediction faled
            new_sp = []
            new_angles = []

            for n, bl_s in enumerate(bl_stats):
                if not bl_out_truth[n]:

                    left = bl_s[0]
                    top = bl_s[1]
                    width = bl_s[2]
                    height = bl_s[3]
                    area = bl_s[4]

                    if area < 3000 * self.config['data']['img_size'] / 1024:
                        continue

                    if width > height:
                        for y in range(top, top + height):
                            if bl_labels[y, left + 1] == n:
                                new_sp.append(np.array([left, y], dtype=np.float64))
                                new_angles.append(0.0)
                                break
                    else:
                        y_direction = 0

                        for x in range(left, left + width):
                            if bl_labels[top + 10, x] == n:
                                y_direction = 1  # -1
                                break
                            elif border_labels[top + 10, x] != 0:
                                y_direction = -1  # -1
                                break

                        if y_direction < 0:  # from bottom up
                            for x in range(left, left + width):
                                if bl_labels[top + height - 1, x] == n:
                                    new_sp.append(np.array([x, top + height - 1], dtype=np.float64))
                                    new_angles.append(math.pi / 2)
                                    break
                        elif y_direction > 0:  # from up to down
                            for x in range(left, left + width):
                                if bl_labels[top + 1, x] == n:
                                    new_sp.append(np.array([x, top + 1], dtype=np.float64))
                                    new_angles.append(-math.pi / 2)
                                    break

            sp_buff = []
            angle_buff = []

            for n in range(len(new_sp)):
                x = new_sp[n][0]
                y = new_sp[n][1]

                d_min = 1e10
                for l in label_list:
                    d = np.sqrt(pow(x - sp_labels[l][0], 2) + pow(y - sp_labels[l][1], 2))
                    if d > 0:
                        d_min = min(d_min, d)
                if d_min > 15:
                    sp_buff.append(new_sp[n])
                    angle_buff.append(new_angles[n])

            new_sp = sp_buff
            new_angles = angle_buff

            if label_list:
                m = max(label_list)
            else:
                m = 0

            new_sp_dict = {l + m + 1: new_sp[l] for l in range(0, len(new_sp))}
            new_angle_dict = {l + m + 1: new_angles[l] for l in range(0, len(new_angles))}

            sp_labels.update(new_sp_dict)
            angles.update(new_angle_dict)
        #####################################################################################

        return sp_labels, ep_labels, angles, label_list

    def extract_baselines(self, image: torch.tensor, image_seg_in, start_points=None, angles=None,
                          with_segmentation=False):
        image = image.unsqueeze(0).to(self.device_lf)
        image_seg_in = image_seg_in.unsqueeze(0).to(self.device_lf)

        if start_points is None:
            with_segmentation = True

        with torch.no_grad():
            if with_segmentation:
                seg_out = self.line_finder_seg(image_seg_in)['out'].detach()
                # seg_out = nn.Sigmoid()(self.line_finder_seg(image_seg_in)['out']).detach()
                seg_out = nn.functional.interpolate(seg_out, size=image.size()[2:], mode='nearest')

                image_seg = torch.cat([image[:, 0:1, :, :], seg_out[:, [self.classes.index('text'),
                                                                        self.classes.index('baselines')], :, :]],
                                      dim=1).detach()

            # if baselines is None extract the start points and angles from the segmentation
            if start_points is None:

                seg_out = nn.Softmax()(seg_out)
                start_points, end_points, angles, label_list = self.extract_start_points_and_angles(
                    seg_out.cpu().detach().numpy())
                label_list = start_points.keys()
                sp_values = torch.tensor(list(start_points.values())).to(self.device_lr)

                # start_points_dict, end_points_dict, angles_dict, label_list = self.extract_start_points_and_angles(
                #     seg_out.cpu().detach().numpy())
                # start_points = []
                # end_points = {}
                # angles = []
                # for l in start_points_dict.keys():w
                #     start_points.append(start_points_dict[l])
                #     angles.append(angles_dict[l])
                #     # if l in end_points.keys():
                #     #     end_points.append(end_points_dict[l])
                # start_points = torch.tensor(start_points)
                # angles = torch.tensor(angles)
                # sp_values = start_points#.to(self.device_lr)
                #
                # label_list = list(range(0, len(start_points)))
            else:
                label_list = list(range(0, len(start_points)))
                # start_points = {l: start_points_list[l] for l in label_list}
                # angles = {l: angles_list[l] for l in label_list}
                sp_values = start_points.to(self.device_lr)
                start_points = {l: start_points[l] for l in label_list}
                angles = {l: angles[l] for l in label_list}
                end_points = {}

            baselines = []
            ep_label_list = end_points.keys()

            for l in label_list:
                sp = torch.tensor(start_points[l]).to(self.device_lr)
                angle = torch.tensor(angles[l]).to(self.device_lr)

                # Compute box size
                box_size = int(get_smallest_distance(sp, sp_values))
                box_size = max(10, min(48, box_size))
                if box_size == 0:
                    box_size = min(10, max(32, get_median_diff(start_points) / 2.0))

                box_size = torch.tensor(box_size).float()

                if l in ep_label_list:
                    ep = torch.tensor(end_points[l]).to(self.device_lr)

                    # TODO: add segmentation
                    if with_segmentation:
                        baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)
                    else:
                        baseline, _, _, _ = self.line_rider(img=image, box_size=box_size, sp=sp, angle_0=angle)

                    baseline[-1] = ep
                else:
                    # TODO: add segmentation
                    if with_segmentation:
                        baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)
                    else:
                        baseline, _, _, _ = self.line_rider(img=image, box_size=box_size, sp=sp, angle_0=angle)

                baselines.append(baseline)

        return baselines
