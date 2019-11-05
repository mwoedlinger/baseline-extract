import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .xml_parser import XMLParser
from ..utils.distances import d2
from ..utils.normalize_baselines import compute_start_and_angle


class DatasetLineFinder(Dataset):
    def __init__(self, inf_type: str, parameters: dict):
        self.input_folder = os.path.join(parameters['input_folder'], inf_type)
        self.inf_type = inf_type
        self.images, self.labels = self.list_files()
        self.max_side = parameters['max_side']
        self.transforms = transforms.Compose([transforms.Resize((self.max_side, self.max_side),#Keep resize?
                                                                interpolation=Image.NEAREST),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])

        self.max_bl_length = 1000
        self.max_bl_no = 2500

    def __len__(self) -> int:
        return len(self.images)

    def get_median_diff(self, start_points):
        diff = []
        N = len(start_points)

        for n in range(0, N-1):
            x1 = start_points[n][0]
            y1 = start_points[n][1]
            x2 = start_points[n+1][0]
            y2 = start_points[n+1][1]
            diff.append(d2(x1, y1, x2, y2))

        return np.median(diff)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        baselines = parser.get_baselines()

        baselines = [[p.get_as_list() for p in bl] for bl in baselines]
        start_points_and_angles = [compute_start_and_angle(baseline=torch.tensor(bl),
                                                           idx=0,
                                                           data_augmentation=False)
                                   for bl in baselines]
        start_points = torch.tensor([bl[0] for bl in baselines])
        box_size = self.get_median_diff(start_points)
        labels = torch.tensor([sa + (torch.tensor(box_size), ) for sa in start_points_and_angles])

        sample = {'image': self.transforms(image),
                  'labels': labels}

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


