import argparse
import json
from src.trainer.trainer import Trainer

def train(config, weights=None):
    t = Trainer(config, weights)
    t.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False, default='config.json')
    args = vars(parser.parse_args())

    with open(args['config'], 'r') as json_file:
        config = json.loads(json_file.read())

    train(config=config, weights=None)
