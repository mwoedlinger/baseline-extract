import argparse
import json
from src.trainer.trainer_line_rider import TrainerLineRider
from src.trainer.trainer_line_finder import TrainerLineFinder

def train(config, model_type, weights=None):
    if model_type == 'line_rider':
        t = TrainerLineRider(config, weights)
    elif model_type == 'line_finder':
        t = TrainerLineFinder(config, weights)

    t.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False, default='config.json')
    parser.add_argument('--model_type', help='line_finder or line_rider', required=False, default='line_rider')
    args = vars(parser.parse_args())

    with open(args['config'], 'r') as json_file:
        config = json.loads(json_file.read())

    train(config=config, model_type=args['model_type'], weights=None)
