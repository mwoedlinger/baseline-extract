import argparse
import json
import os
from src.trainer.trainer_line_rider import TrainerLineRider


def test(config, weights=None):
    t = TrainerLineRider(config, weights)
    t.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False)
    args = vars(parser.parse_args())

    config_file = args['config']
    if config_file is None:
        config_file = 'config_' + args['model_type'] + '.json'

    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())
    trained_model = os.path.join(config['output_folder'], 'line_rider', 'line_rider_' + config['exp_name'] + '.pt')

    test(config=config, weights=trained_model)
