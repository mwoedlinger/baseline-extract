import argparse
import json
from src.trainer.trainer import Trainer

def train(config, weights=None):
    t = Trainer(config, weights=None)
    t.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False, default='config.json')
    args = vars(parser.parse_args())

    with open(args['config']) as json_file:
        config = json.load(json_file)

    train(config=config, weights=None)
