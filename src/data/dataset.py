import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .xml_parser import XMLParser


class BaselineDataset(Dataset):
    """
    A dataset that generates data for a pytorch model.
    """
    def __init__(self, inf_type: str, parameters: dict):
        self.input_folder = os.path.join(parameters['input_folder'], inf_type)
        self.inf_type = inf_type
        self.images, self.labels = self.list_files()
        self.max_side = parameters['max_side']
        self.transforms = transforms.Compose([transforms.Resize((self.max_side, self.max_side),
                                                                interpolation=Image.NEAREST),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        baselines = parser.get_baselines()

        start_points = []
        start_angles = []

        for bl in baselines:
            p0 = bl[0].get_as_list()
            p1 = bl[1].get_as_list()

            angle = np.arctan((p0[1]-p1[1])/(p0[0]-p1[0]))
            #TODO: ^^ make sure that the angle doesn't flip for vertical baselines

            start_points.append(p0)
            start_angles.append(angle)

        baselines = [[p.get_as_list() for p in bl] for bl in baselines]

        sample = {'image': self.transforms(image),
                  'baselines': torch.tensor(baselines),
                  'start_points': torch.tensor(start_points),
                  'start_angles': torch.tensor(start_angles)}

        return sample

    def list_files(self) -> tuple:
        """
        Get list of all image files in the folders 'images' and 'labels'
        :return:
        """
        xml_dir = os.path.join(self.input_folder, 'page')

        for root, directories, filenames in os.walk(self.input_folder):
            xml_list = [os.path.join(xml_dir, f) for f in filenames]

        basename_list = [os.path.basename(f).split('.')[0] for f in xml_list]
        image_list = [os.path.join(self.input_folder, f+'.jpg') for f in basename_list]

        return image_list, xml_list


