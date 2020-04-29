import argparse
import json
import os
from src.trainer.trainer_line_rider import TrainerLineRider


def train(config, model_type, weights=None):
    if model_type == 'line_rider':
        t = TrainerLineRider(config, weights)
    elif model_type == 'line_finder':
        raise NotImplementedError

    t.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--model_type', help='line_finder or line_rider', required=False, default='line_rider')
    parser.add_argument('--config', help='The config file.', required=False)
    args = vars(parser.parse_args())

    config_file = args['config']
    if config_file is None:
        config_file = 'config_' + args['model_type'] + '.json'

    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())
    trained_model = os.path.join(config['output_folder'], 'line_rider', 'line_rider_' + config['exp_name'] + '.pt')

    train(config=config, model_type=args['model_type'], weights=trained_model)
