import os
import PIL
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from .xml_parser import XMLParser


class DatasetLineRider(Dataset):
    """
    Extracts baseline coordinates from page xml files. Returns the original image together with the baselines and
    their respective lengths. Resizes to the max_side parameter given in the config dict 'parameters'.
    """
    def __init__(self, inf_type: str, parameters: dict):
        self.input_folder = os.path.join(parameters['input_folder'], inf_type)
        self.inf_type = inf_type
        self.images, self.labels = self.list_files()
        self.min_side = parameters['min_side']
        self.transforms = transforms.Compose([transforms.Resize((self.min_side, self.min_side),
                                                                interpolation=PIL.Image.NEAREST),
                                              transforms.Grayscale(num_output_channels=3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                                   std=[0.2174, 0.2115, 0.1989])
                                              ])
        self.seg_transforms = transforms.Compose([transforms.Resize(self.min_side,
                                                                interpolation=PIL.Image.NEAREST),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                                   std=[0.2174, 0.2115, 0.1989])
                                              ])

        self.max_bl_length = 1000
        self.max_bl_no = 2500

        random.seed(42)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = PIL.Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.min_side)
        baselines = parser.get_baselines()
        #TODO: add data augmentation: negative colors? also add segmentation!

        baselines = [[p.get_as_list() for p in bl] for bl in baselines]
        # bl_length is a list that contains the length of every baseline. This will be returned to make it
        # easier to remove the padding.
        bl_lengths = [len(bl) for bl in baselines]

        # We assume a maximum baseline length of self.max_bl_length and a maximum number of baselines of self.max_bl_no.
        # Everything below that will be padded such that the baselines returned will be saved in a
        # (self.max_bl_no, self.max_bl_length, 2) tensor and bl_length will be stored in (self.max_bl_no) tensors.
        bl_lengths_padded = bl_lengths + [-1]*(self.max_bl_no - len(bl_lengths))
        baselines_padded_linewise = [b + [[-1, -1]]*(self.max_bl_length-len(b)) for b in baselines]
        baselines_padded = baselines_padded_linewise + [[[-1, -1]]*self.max_bl_length]*(self.max_bl_no - len(baselines_padded_linewise))

        sample = {'image': self.transforms(image),
                  'seg_image': self.seg_transforms(image),
                  'baselines': torch.tensor(baselines_padded),
                  'bl_lengths': torch.tensor(bl_lengths_padded)}

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