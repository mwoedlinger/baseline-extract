import numpy as np
import cv2
import torch
from ..segmentation.gcn_model import GCN
from ..utils.distances import get_smallest_distance, get_median_diff


class LineDetector:
    """
    Loads the line rider model for inference.
    """
    def __init__(self, config: dict):
        line_rider_weights = config['line_rider']['weights']
        line_finder_weights = config['line_finder']['weights']
        self.device_lr = torch.device('cuda:' + str(config['line_rider']['device']) if torch.cuda.is_available() else 'cpu')
        self.device_lf = torch.device('cuda:' + str(config['line_finder']['device']) if torch.cuda.is_available() else 'cpu')
        num_classes = config['line_finder']['num_classes']
        backbone = config['line_finder']['backbone']

        print('## Load Line Rider:')
        self.line_rider = self.load_line_rider_model(line_rider_weights, self.device_lr)
        print('## Load Line Finder:')
        self.line_finder_seg = self.load_line_finder_model(line_finder_weights, self.device_lf, num_classes, backbone)
        print('## Loaded!')


    def load_line_rider_model(self, weights, device):
        model = torch.load(weights, map_location=device)
        model.to(device)
        model.eval()
        model.device = device

        return model

    def load_line_finder_model(self, weights, device, num_classes, backbone):
        seg_model = GCN(n_classes=num_classes, resnet_depth=backbone)
        seg_model.load_state_dict(torch.load(weights, map_location=device))
        seg_model.to(device)
        seg_model.eval()

        return seg_model

    def segmentation_postprocessing(self, array: np.array, sigma, threshold, morph_close_size, erode_size):
        out = cv2.GaussianBlur(array, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        out = (out > threshold) * 1.0
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, (morph_close_size, morph_close_size))
        out = cv2.erode(out, (erode_size, erode_size), iterations=1)

        return out

    def extract_start_points_and_angles(self, seg: np.array) -> tuple:
        segmentation = np.transpose(seg[0], (1, 2, 0))

        probs_start_points = segmentation[:, :, 2]
        probs_end_points = segmentation[:, :, 3]
        probs_baselines = segmentation[:, :, 0]

        # Postprozessing parameters
        sigma = 0.3
        threshold = 0.99
        morph_close_size = 3
        erode_size = 3

        # Extract start points
        sp = self.segmentation_postprocessing(probs_start_points, sigma, threshold, morph_close_size, erode_size)
        _, _, _, start_points = cv2.connectedComponentsWithStats(sp.astype(np.uint8))

        # Extract end points
        ep = self.segmentation_postprocessing(probs_end_points, sigma, threshold, morph_close_size, erode_size)
        _, _, _, end_points = cv2.connectedComponentsWithStats(ep.astype(np.uint8))

        # Compute the angles and match start and end points
        probs_sum = probs_start_points + probs_end_points + probs_baselines
        ps = cv2.GaussianBlur(probs_sum, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        ps = (ps > threshold) * 1.0
        ps = cv2.morphologyEx(ps, cv2.MORPH_OPEN, (3, 3))

        _, labels, stats, _ = cv2.connectedComponentsWithStats(ps.astype(np.uint8))

        # stats is a matrix where for every label l the vector stats[l] is given by:
        # [leftmost (x) coordinate, topmost (y) coordinate, width of bounding box, height of bouinding box, area]
        # see https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python

        # Assign labels to the start and end points
        # [1:] because the background is also a component
        sp_labels = {labels[(int(p[1]), int(p[0]))]: p for p in start_points[1:]}
        ep_labels = {labels[(int(p[1]), int(p[0]))]: p for p in end_points[1:]}

        label_list = sp_labels.keys()
        angles = {l: np.arctan(stats[l][3] / stats[l][2]) for l in label_list}

        return sp_labels, ep_labels, angles, label_list

    def extract_baselines(self, image: torch.tensor):
        image = image.unsqueeze(0).to(self.device_lf)
        seg_out = self.line_finder_seg(image)[0]

        image_seg = torch.cat([image[:, 0:1, :, :], seg_out[:, [0, 1], :, :]], dim=1).detach()

        start_points, end_points, angles, label_list = self.extract_start_points_and_angles(seg_out.cpu().detach().numpy())

        sp_values = torch.tensor(list(start_points.values())).to(self.device_lr)

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

            box_size = torch.tensor(box_size).double()

            if l in ep_label_list:
                ep = torch.tensor(end_points[l]).to(self.device_lr)

                baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)

                baseline[-1] = ep
            else:
                baseline, _, _, _ = self.line_rider(img=image_seg, box_size=box_size, sp=sp, angle_0=angle)

            baselines.append(baseline)

        return baselines






