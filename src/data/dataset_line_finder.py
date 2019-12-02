import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .xml_parser import XMLParser
from ..utils.distances import d2
from ..utils.normalize_baselines import compute_start_and_angle
from ..segmentation.gcn_model import GCN
import time


class DatasetLineFinder(Dataset):
    def __init__(self, inf_type: str, parameters: dict, random_seed: int=42):
        random.seed(random_seed)
        self.input_folder = os.path.join(parameters['input_folder'], inf_type)
        self.inf_type = inf_type
        self.images, self.labels = self.list_files()
        self.max_side = parameters['max_side']
        self.patch_size = parameters['crop_size']
        self.transform_grayscale = transforms.Grayscale(num_output_channels=1)
        self.transform_to_pil = transforms.ToPILImage()
        self.transform_normalize = transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                        std=[0.2174, 0.2115, 0.1989])
        self.transform_resize = transforms.Resize((self.max_side, self.max_side), interpolation=Image.NEAREST)
        self.transform_to_tensor = transforms.ToTensor()

        self.max_bl_length = 1000
        self.max_bl_no = 2500

    def __len__(self) -> int:
        return len(self.images)

    def get_median_diff(self, start_points):
        diff = []
        N = len(start_points)

        if N > 0:
            for n in range(0, N-1):
                x1 = start_points[n][0]
                y1 = start_points[n][1]
                x2 = start_points[n+1][0]
                y2 = start_points[n+1][1]
                diff.append(d2(x1, y1, x2, y2))

            return np.median(diff)
        else:
            return -1

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        baselines = parser.get_baselines()
        baselines = [[p.get_as_list() for p in bl] for bl in baselines]


        # Resize image:
        image = self.transform_resize(image)
        image = self.transform_normalize(self.transform_to_tensor(image))

        # image = self.transform_to_pil(image)
        #
        # patch_diff = self.patch_size/2#self.max_side-self.patch_size
        #
        # Crop patch:
        # i = random.randint(-patch_diff/2, self.max_side - self.patch_size + patch_diff/2)
        # j = random.randint(-patch_diff/2, self.max_side - self.patch_size + patch_diff/2)
        # height = self.patch_size
        # width = self.patch_size
        # image = transforms.functional.crop(img=image, i=i, j=j, h=height, w=width)
        #
        # baselines = [[[p[0]-j, p[1]-i] for p in bl] for bl in baselines]


        start_points_and_angles = [compute_start_and_angle(baseline=torch.tensor(bl),
                                                           idx=0,
                                                           data_augmentation=False)
                                   for bl in baselines]
        labels = [l for l in start_points_and_angles if 0 <= l[0] < self.patch_size and 0 <= l[1] < self.patch_size]
        labels = torch.tensor(labels)
        labels_exp = torch.cat([labels, torch.tensor([[-1.0, -1.0, -1.0]]*(10000-len(labels)))], dim=0)

        sample = {'image': image,
                  'label': labels_exp,
                  'label_len': len(labels)
                  }

        return sample

    def list_files(self) -> tuple:
        """
        Get lists of all image and xml files in the folders 'images' and 'labels'
        :return: A tuple of lists: (image_list, xml_list)
        """
        xml_dir = os.path.join(self.input_folder, 'page')

        for root, directories, filenames in os.walk(self.input_folder):
            xml_list = [os.path.join(xml_dir, f) for f in filenames]

        basename_list = [os.path.basename(f).split('.')[0] for f in xml_list]
        image_list = [os.path.join(self.input_folder, f+'.jpg') for f in basename_list]

        return image_list, xml_list


