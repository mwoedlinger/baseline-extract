import argparse
import json
from src.inference.models import LineDetector
from src.data.dataset_inference import DatasetInference

def predict(config):
    detector = LineDetector(config)
    dataset = DatasetInference(config['data']['img_size'], config['data']['input_folder'])

    #TODO: write loop



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--model_type', help='line_finder or line_rider', required=False, default='line_rider')
    parser.add_argument('--config', help='The config file.', required=False, default='config_inference.json')
    args = vars(parser.parse_args())

    config_file = args['config']
    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())

    predict(config=config)
