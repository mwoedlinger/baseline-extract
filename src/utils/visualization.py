import torch
import torchvision
import numpy as np
import cv2
from .distances import get_median_diff


def draw_baselines(image: torch.tensor, baselines: torch.tensor, idx=-1):
    """
    Creates a visualization of the model output. It draws points every baseline point and connects them with lines.
    Returns the resulting image as a torch tensor.
    :param image: The input image, assumes that is is normalized with imageNet parameters.
    :param baselines: The predicted baselines as a torch tensor
    :param idx: (optional) if set to a specific index, colors this baseline in a different color
    :return: The resulting image
    """
    h = image.shape[1]
    w = image.shape[2]

    img_cv1 = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_cv1 = std * img_cv1 + mean
    img_cv1 = np.clip(img_cv1, 0, 1)

    img_cv2 = np.zeros((h, w, 3), dtype=np.float32)

    baselines_n = baselines

    for N, bl in enumerate(baselines_n):
        for k, p in enumerate(bl):
            if k == 0:
                coords = (int(p[0]), int(p[1]))
                c_last = coords

                cv2.circle(img_cv2, coords, 3, (0, 0, 1.0), thickness=3)
                continue
            else:
                coords = (int(p[0]), int(p[1]))
                if N == idx:
                    cv2.line(img_cv2, c_last, coords, (0, 1.0, 0), thickness=2)
                else:
                    cv2.line(img_cv2, c_last, coords, (1.0, 0, 0), thickness=2)

                cv2.circle(img_cv2, coords, 3, (0, 0, 1.0), thickness=3)
                c_last = coords

    img_cv2 = img_cv2.astype(np.float64)
    comb = cv2.addWeighted(img_cv2, 0.7, img_cv1, 0.3, 0)
    comb = torchvision.transforms.ToTensor()(comb).float()

    return comb

def draw_start_points(image: torch.tensor, label: torch.tensor, true_label: torch.tensor=None):
    """
    Creates a visualization of the model output. For every start point of every baseline it draws a circle with radius
    box_size and a line with the length 2*box_size and the predicted angle
    :param image: The input image, assumes that is is normalized with imageNet parameters.
    :param label: The predicted labels
    :return: The resulting image
    """
    img = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    sp = label[:, 0:2].cpu()
    if true_label is not None:
        if len(true_label) < 1:
            true_label = None
        else:
            sp_true = true_label[:, 0:2].cpu()
    angles = label[:, 2].cpu()
    box_size = get_median_diff(sp)
    confidences = label[:, -1].cpu()

    labels_img = np.zeros(img.shape)
    thickness = 2

    for k, s in enumerate(sp):
        if confidences[k].item() < 0.01:
            continue
        else:
            ccolor = confidences[k].item()
            if box_size <= 0:
                box_size = -np.abs(box_size[k])-0.0001
            x = int(s[0].item())
            y = int(s[1].item())
            cv2.circle(labels_img, (x, y), box_size, (ccolor, 0, 0), thickness)
            cv2.line(labels_img, (x, y),
                     (x + 2 * box_size * np.cos(angles[k]), y - 2 * box_size * np.sin(angles[k])),
                     (0, ccolor, 0), thickness)

    if true_label is not None:
        for s in sp_true:
            # Draw GT start points
            x = int(s[0].item())
            y = int(s[1].item())
            cv2.circle(labels_img, (x, y), 5, (0, 0, 1.0), thickness)

    comb = cv2.addWeighted(img, 0.5, labels_img, 0.5, 0)
    comb = torchvision.transforms.ToTensor()(comb).float()

    return comb