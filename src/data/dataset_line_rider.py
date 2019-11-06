import os
from PIL import Image
import torch
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
        # TODO: implement data augmentation: horizontal flip:
        # Flip the entire input image horizontally. Also 'flip' the coordinates of the baselines => inverse the order
        # (baselines should be read from left to right)
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        baselines = parser.get_baselines()

        baselines = [[p.get_as_list() for p in bl] for bl in baselines]
        # bl_length is a list that contains the length of every baseline. This will be returned to make it
        # easier to remove the padding.
        bl_lengths = [len(bl) for bl in baselines]

        # We assume a maximum baseline length of self.max_bl_length and a maximum number of baselines of self.max_bl_no.
        # Everything below that will be padded such that the baselines returned will be saved in a
        # (self.max_bl_no, self.max_bl_length, 2) tensor and bl_length will be stored in (self.max_bl_no) tensors.
        bl_lengths_padded = bl_lengths + [-1]*(self.max_bl_no - len(bl_lengths))
        baselines_padded_linewise = [b + [[-1,-1]]*(self.max_bl_length-len(b)) for b in baselines]
        baselines_padded = baselines_padded_linewise + [[[-1, -1]]*self.max_bl_length]*(self.max_bl_no - len(baselines_padded_linewise))

        sample = {'image': self.transforms(image),
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


