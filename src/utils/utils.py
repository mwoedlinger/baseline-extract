from torchvision import transforms
import PIL
import torch
import tqdm
import os

def load_class_dict(label_file: str):
    """
    Reads the class label file and returns a tuple consisting of a list of classes, a list of colors
    and a dictionary {class: color}
    :param label_file: The text file containg the class names and colors
    :return: A tuple of classes, colors and a dicitonary {class: color}
    """
    with open(label_file, 'r') as tf:
        text = tf.readlines()
    classes_file = [t.strip() for t in text]

    classes = [l.split(';')[0] for l in classes_file]
    colors = [[int(l.split(';')[n]) for n in range(1, len(l.split(';')))] for l in classes_file]
    class_dict = {classes[n]: colors[n] for n in range(len(classes))}

    return classes, colors, class_dict

def compute_dataset_mean(img_folder):
    file_list = []
    for root, dirs, files in os.walk(img_folder):
        file_list += [os.path.join(root, f) for f in files]

    running_mean = 0
    running_sum = 0
    counter = 0
    t = transforms.ToTensor()

    for f in tqdm.tqdm(file_list):
        img = PIL.Image.open(f)
        img_t = t(img)

        # compute mean
        mean = img_t.mean([1, 2])
        running_mean += mean

        # compute std
        img_t = (img_t - mean.unsqueeze(1).unsqueeze(1)) ** 2
        running_sum += img_t.sum([1, 2])
        counter += torch.numel(img_t[0])

        mean = running_mean/len(file_list)
        std = torch.sqrt(running_sum/counter)

    return mean, std # [0.7219, 0.6874, 0.6260], [0.2174, 0.2115, 0.1989]


def create_prediction_string(baselines: list, width: int, height: int, img_size: int):
    w_ratio = width/img_size
    h_ratio = height/img_size

    prediction_string_list = []

    for bl in baselines:
        bl_list = []

        for p in bl:
            x = int(p[0].item() * w_ratio)
            y = int(p[1].item() * h_ratio)
            bl_list.append(str(x) + ',' + str(y))

        prediction_string_list.append(';'.join(bl_list))

    prediction_string = '\n'.join(prediction_string_list)

    return prediction_string

def load_class_dict(label_file: str):
    """
    Reads the class label file and returns a tuple consisting of a list of classes, a list of colors
    and a dictionary {class: color}
    :param label_file: The text file containg the class names and colors
    :return: A tuple of classes, colors and a dicitonary {class: color}
    """
    with open(label_file, 'r') as tf:
        text = tf.readlines()
    classes_file = [t.strip() for t in text]

    classes = [l.split(';')[0] for l in classes_file]
    colors = [[int(l.split(';')[n]) for n in range(1, len(l.split(';')))] for l in classes_file]
    class_dict = {classes[n]: colors[n] for n in range(len(classes))}

    return classes, colors, class_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']