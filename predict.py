import argparse
import json
import tqdm
import os
from src.inference.models import LineDetector
from src.data.dataset_inference import DatasetInference
from src.utils.utils import create_prediction_string, load_class_dict

def create_lst_files(gt_folder = os.path.join('data', 'evaluation_java_app', 'gt'),
                     pred_folder = os.path.join('data', 'pred')):
    folders = [gt_folder, pred_folder]
    for folder in folders:
        file_list = []

        for root, _, files in os.walk(folder):
            file_list += [os.path.join(root, f) for f in files]

        print('Found {} files in {} folder.'.format(len(file_list), folder))

        text_string = '\n'.join(file_list)
        filename = os.path.join('data', os.path.basename(folder) + '.lst')
        with open(filename, 'w') as txt_file:
            txt_file.writelines(text_string)
            print('Created {}'.format(filename))

def predict(config):
    detector = LineDetector(config)
    dataset = DatasetInference(config['data']['img_size'],
                               config['data']['input_folder'],
                               get_GT_start_points=config['line_finder']['use_GT_sp_and_angles'])

    output_folder = config['output_folder']
    if not os.path.isdir(os.path.join(output_folder)):
        os.mkdir(output_folder)

    print('## Start prediction:')
    for sample in tqdm.tqdm(dataset):
        img = sample['image']
        seg_img = sample['seg_image']
        filename = sample['filename']
        width = sample['width']
        height = sample['height']

        if config['line_finder']['use_GT_sp_and_angles']:
            sp = sample['start_points']
            angles = sample['angles']
            baselines = detector.extract_baselines(img, seg_img, sp, angles,
                                                   with_segmentation=config['line_rider']['with_segmentation'])
        else:
            baselines = detector.extract_baselines(img, seg_img)
        bl_string = create_prediction_string(baselines, width, height, config['data']['img_size'])

        text_name = os.path.basename(filename).split('.')[0]+'.txt'
        text_name.replace('data', '').replace('/', '\\')
        with open(os.path.join(output_folder, text_name), 'w') as txt_file:
            txt_file.writelines(bl_string)

    print('## Create lst files.')
    create_lst_files(pred_folder=output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model..')
    parser.add_argument('--config', help='The config file.', required=False, default='config_inference.json')
    args = vars(parser.parse_args())

    config_file = args['config']
    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())

    predict(config=config)
