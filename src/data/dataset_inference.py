import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import PIL


class DatasetInference(Dataset):
    def __init__(self, img_size: int, input_folder: str):
        self.input_folder = input_folder
        self.images = self.list_files()
        self.img_size = img_size
        self.transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size),
                                                                interpolation=PIL.Image.NEAREST),
                                              transforms.Grayscale(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.6784], std=[0.2092])
                                              ]),

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        image = self.transforms(image)

        return image

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

        return image_list


