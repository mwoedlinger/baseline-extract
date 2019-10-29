import torch
import torchvision
import numpy as np
import cv2
from .normalize_baselines import normalize_baselines


def draw_baselines(image: torch.tensor, baselines: torch.tensor, bl_lengths: torch.tensor, idx=-1):
    h = image.shape[1]
    w = image.shape[2]

    img_cv1 = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_cv1 = std * img_cv1 + mean
    img_cv1 = np.clip(img_cv1, 0, 1)

    img_cv2 = np.zeros((h, w, 3), dtype=np.float32)

    # number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])
    # baselines_n = [normalize_baselines(b[:bl_lengths[n]], box_size, device=device)
    #                for n, b in enumerate(baselines[0:number_of_baselines])]
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
    # comb = cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
    comb = torchvision.transforms.ToTensor()(comb).float()

    return comb