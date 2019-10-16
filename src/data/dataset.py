import os
from torch.utils.data import Dataset
from PIL import Image
from .xml_parser import XMLParser


class baselineDataset(Dataset):
    """
    A dataset that generates data for a pytorch model.
    """
    def __init__(self, input_folder: str, inf_type: str, parameters: dict):
        self.input_folder = input_folder
        self.inf_type = inf_type
        self.images, self.labels = self.list_files()
        self.max_side = parameters['max_side']

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        parser = XMLParser(self.labels[idx])
        parser.scale(self.max_side)
        label = parser.get_baselines()

        sample = {'image': image, 'label': label}

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


