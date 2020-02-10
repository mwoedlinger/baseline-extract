import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from ..utils.normalize_baselines import compute_start_and_angle
from .xml_parser_inference import XMLParserInference
import PIL


class DatasetInference(Dataset):
    def __init__(self, img_size: int, input_folder: str, get_GT_start_points: bool):
        self.input_folder = input_folder
        self.GT_sp = get_GT_start_points
        self.images, self.xml_files = self.list_files()
        self.img_size = img_size
        self.transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size),
                                                                interpolation=PIL.Image.NEAREST),
                                              transforms.Grayscale(num_output_channels=3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.6784], std=[0.2092])
                                              ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])

        width, height = image.size

        image = self.transforms(image)

        if self.GT_sp:
            parser = XMLParserInference(self.xml_files[idx])
            parser.scale(self.img_size)
            baselines = parser.get_baselines()
            baselines = [[p.get_as_list() for p in bl] for bl in baselines]

            start_points = []
            angles = []

            for bl in baselines:
                start_points.append(bl[0])
                bl_t = torch.tensor(bl)
                _, _, angle = compute_start_and_angle(bl_t, idx=0)
                angles.append(angle)


            sample = {'image': image,
                      'filename': self.images[idx],
                      'width': width,
                      'height': height,
                      'start_points': torch.tensor(start_points),
                      'angles': torch.tensor(angles)}
        else:
            sample = {'image': image,
                      'filename': self.images[idx],
                      'width': width,
                      'height': height}

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


