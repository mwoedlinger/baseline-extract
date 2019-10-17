import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
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

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        baselines = parser.get_baselines()

        start_points = []
        start_angles = []

        for bl in label:
            p0 = bl[0].get_as_list()
            p1 = bl[1].get_as_list()

            angle = np.arctan((p0[1]-p1[1])/(p0[0]-p0[0]))

            start_points.append(p0)
            start_angles.append(angle)

        sample = {'image': image, 'baselines': baselines, 'start_points': start_points, 'start_angles': start_angles}

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


