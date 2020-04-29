import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from ..segmentation.gcn_model import GCN
from ..utils.distances import get_smallest_distance, get_median_diff
from ..utils.utils import load_class_dict
from .inference_utils import *


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

        print('## Load Line Rider:')
        self.line_rider = self.load_line_rider_model(line_rider_weights, self.device_lr)
        if config['line_rider']['with_segmentation']:
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

            else:
                label_list = list(range(0, len(start_points)))
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

                    if with_segmentation:
                        baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)
                    else:
                        baseline, _, _, _ = self.line_rider(img=image, box_size=box_size, sp=sp, angle_0=angle)

                    baseline[-1] = ep
                else:
                    if with_segmentation:
                        baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)
                    else:
                        baseline, _, _, _ = self.line_rider(img=image, box_size=box_size, sp=sp, angle_0=angle)

                baselines.append(baseline)

        return baselines
