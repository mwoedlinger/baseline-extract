import argparse
import json
import tqdm
import os
from src.inference.models import LineDetector
from src.data.dataset_inference import DatasetInference
from src.utils.utils import create_prediction_string


def predict(config):
    detector = LineDetector(config)
    dataset = DatasetInference(config['data']['img_size'], config['data']['input_folder'])

    output_folder = config['output_folder']
    if not os.path.isdir(os.path.join(output_folder)):
        os.mkdir(output_folder)

    print('## Start prediction:')
    for sample in tqdm.tqdm(dataset):
        img = sample['image']
        filename = sample['filename']
        width = sample['width']
        height = sample['height']

        baselines = detector.extract_baselines(img)
        bl_string = create_prediction_string(baselines, width, height, config['data']['img_size'])

        text_name = os.path.basename(filename).split('.')[0]+'.txt'
        with open(os.path.join(output_folder, text_name), 'w') as txt_file:
            txt_file.writelines(bl_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False, default='config_inference.json')
    args = vars(parser.parse_args())

    config_file = args['config']
    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())

    predict(config=config)
