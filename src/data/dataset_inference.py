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

        for root, directories, filenames in os.walk(self.input_folder):
            if 'page' not in root:
                image_list = [os.path.join(root, f) for f in filenames]
        image_list.sort()

        return image_list


