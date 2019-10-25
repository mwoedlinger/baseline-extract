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

        self.max_bl_length = 1000
        self.max_bl_no = 2500

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

            if (np.abs(p0[0]-p1[0]) < 0.00001):
                angle = np.pi/2
            else:
                angle = np.arctan((p0[1]-p1[1])/(p0[0]-p1[0]))
            #TODO: ^ make sure that the angle doesn't flip for vertical baselines

            start_points.append(p0)
            start_angles.append(angle)

        baselines = [[p.get_as_list() for p in bl] for bl in baselines]
        # bl_length is a list that contains the length of every baseline. This will be returned to make it
        # easier to remove the padding.
        bl_lengths = [len(bl) for bl in baselines]

        # We assume a maximum baseline length of self.max_bl_length and a maximum number of baselines of self.max_bl_no.
        # Everything below that will be padded such that the baselines returned will be saved in a
        # (self.max_bl_no, self.max_bl_length, 2) tensor and start_points, start_angles and bl_length will be stored in
        # (self.max_bl_no) tensors.
        bl_lengths_padded = bl_lengths + [-1]*(self.max_bl_no - len(bl_lengths))
        start_angles_padded = start_angles + [-1]*(self.max_bl_no - len(bl_lengths))
        start_points_padded = start_points + [[-1,-1]]*(self.max_bl_no - len(bl_lengths))
        baselines_padded_linewise = [b + [[-1,-1]]*(self.max_bl_length-len(b)) for b in baselines]
        baselines_padded = baselines_padded_linewise + [[[-1,-1]]*self.max_bl_length]*(self.max_bl_no - len(baselines_padded_linewise))

        sample = {'image': self.transforms(image),
                  'baselines': torch.tensor(baselines_padded),
                  'bl_lengths': torch.tensor(bl_lengths_padded),
                  'start_points': torch.tensor(start_points_padded),
                  'start_angles': torch.tensor(start_angles_padded)}

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

        return image_list[0:10], xml_list[0:10]


