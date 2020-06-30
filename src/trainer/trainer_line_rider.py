import os
import time
import copy
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..data.dataset_line_rider import DatasetLineRider
from ..model.loss import prepare_data_for_loss
from ..utils.visualization import draw_baselines
from ..model.line_rider import LineRider
from ..segmentation.gcn_model import GCN
from ..utils.distances import get_smallest_distance
from ..utils.utils import get_lr, load_class_dict


class TrainerLineRider:
    """
    Performs the training of the LineRider network.
    """

    def __init__(self, config, weights=None):
        self.exp_name = 'line_rider_' + config['exp_name']
        self.log_dir = config['log_dir']
        self.output_folder = config['output_folder']
        random.seed(config['random_seed'])

        self.lr = config['lr']
        self.gpu = config['gpu']
        self.with_seg = config['with_seg']
        self.seg_gpu = config['segmentation_gpu']
        self.epochs = config['epochs']
        self.eval_epoch = config['eval_epoch']
        self.parameters = config['data']
        self.min_side = config['data']['min_side']
        self.reset_idx = config['reset_idx_start']

        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')
        self.seg_device = torch.device('cuda:' + str(self.seg_gpu) if torch.cuda.is_available() else 'cpu')
        self.model, self.criterion_bl, self.criterion_end, self.criterion_length = self.get_model(weights)

        self.optimizer, self.scheduler = self.get_optimizer()
        self.dataloaders = self.get_dataloaders()

        # If with_seg is set to True load the segmentation model
        if self.with_seg:
            print('## Loading segmentation model')
            self.classes, _, _ = load_class_dict(config['segmentation_class_file'])
            self.num_classes = len(self.classes)
            self.segmentation_weights = config['segmentation_weights']

            self.seg_model = GCN(n_classes=self.num_classes, resnet_depth=50)
            self.seg_model.load_state_dict(torch.load(self.segmentation_weights, map_location=self.seg_device))
            self.seg_model.to(self.seg_device)
            self.seg_model.eval()

        print('\n## Trainer settings:')
        print('exp name:    {}\n'
              'lr:          {}\n'
              'image size : {}\n'
              'gpu:         {}\n'
              'epochs:      {}\n'
              'log dir:     {}\n'
              'with seg:    {}\n'.format(self.exp_name, self.lr, self.min_side, self.gpu,
                                         self.epochs, self.log_dir, self.with_seg))
        if self.with_seg:
            print('seg_gpu:     {}\n'.format(self.seg_device))

    def get_model(self, weights):
        """
        Load the correct model,
        :return: a list containing the model and the loss functions
        """

        if not weights:
            model_ft = LineRider(device=self.device)
        else:
            print('## Loading pretrained model {}'.format(weights))
            model_ft = torch.load(weights, map_location=self.device)
            model_ft.device = self.device

        model_ft.to(self.device)
        criterion_bl = torch.nn.MSELoss()
        criterion_end = torch.nn.BCELoss()
        criterion_length = torch.nn.MSELoss()

        return [model_ft, criterion_bl, criterion_end, criterion_length]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        # Only used for SGD
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train and eval.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': True, 'eval': False}

        image_datasets = {inf_type: DatasetLineRider(inf_type=inf_type,
                                                     parameters=self.parameters)
                          for inf_type in ['train', 'eval']}

        dataloaders = {inf_type: torch.utils.data.DataLoader(image_datasets[inf_type],
                                                             batch_size=1,
                                                             shuffle=shuffle[inf_type],
                                                             num_workers=1)
                       for inf_type in ['train', 'eval']}

        return dataloaders

    def _train_epoch(self, steps, writer):
        """
        Train the model for one epoch. Returns a a tuple containing the loss and the current step counter.
        :param steps: starting point for step counter. The function returns the updated steps.
        :param epoch: the current epoch number
        :param writer: Tensorboard writer
        :return: A tuple containing the loss and the current step counter.
        """
        self.model.train()
        self.model.data_augmentation = True

        tensorboard_img_steps = 99
        reset_counter_steps = 699

        running_loss = 0
        running_bl_loss = 0
        running_end_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train'], dynamic_ncols=True):
            if steps % reset_counter_steps == reset_counter_steps-1:
                if self.reset_idx < 8:
                    self.reset_idx += 1

            image = batch['image'].to(self.device)
            baselines = batch['baselines'][0].float().to(self.device)
            bl_lengths = batch['bl_lengths'][0].to(self.device)

            # if self.with_seg is true the model performs a segmentation and uses the segmentation information
            # for computing the baselines.
            if self.with_seg:
                seg_image = batch['seg_image'].to(self.seg_device)
                with torch.no_grad():
                    # seg_out = nn.Softmax(dim=0)(self.seg_model(seg_image)['out']).detach().to(self.device)
                    seg_out = self.seg_model(seg_image)['out'].detach().to(self.device)
                    seg_out = nn.functional.interpolate(seg_out, size=image.size()[2:], mode='nearest')
                    seg_out = nn.Softmax()(seg_out)
                image = torch.cat([image[:, 0:1, :, :], seg_out[:, [self.classes.index('text'),
                                                                    self.classes.index('baselines')], :, :]], dim=1).detach()

                # with torch.no_grad():
                #     seg_out = nn.Softmax()(self.seg_model(image.to(self.seg_device))['out']).detach().to(self.device)
                # image = torch.cat([image[:, 0:1, :, :], seg_out[:, [self.classes.index('end_points'), self.classes.index('baselines')], :, :]], dim=1).detach()

            steps += 1

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])
            start_points = torch.tensor([[bl[0, 0], bl[0, 1]] for bl in baselines[0:number_of_baselines]])

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                pred_list = []

            with torch.set_grad_enabled(True):
                # loss = 0
                # loss_end = 0
                for n in range(number_of_baselines):

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Compute box size
                    box_size = int(get_smallest_distance(start_points[n], start_points))
                    box_size = max(10, min(48, box_size)) + random.randint(-3, 10)#48
                    # +5: Larger box sizes are more difficult => regularization

                    gt_baseline = baselines[n][:bl_lengths[n]]

                    pred_baseline, pred_end, _ = self.model(img=image, box_size=box_size, baseline=gt_baseline,
                                                            reset_idx=self.reset_idx)
                    if not pred_baseline.requires_grad:
                        continue

                    # Every tensorboard_img_steps steps save the result to tensorboard:
                    if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                        pred_list.append(pred_baseline)

                    baseline_label, end_label = prepare_data_for_loss(gt_baseline, pred_baseline)
                    baseline_label = baseline_label.to(self.device)
                    end_label = end_label.to(self.device)

                    # The loss is a combination of the regression loss for prediction of the baseline coordinates,
                    # the length of the last baseline segment and the classification loss for the prediction of
                    # the baseline end.
                    loss_bl = self.criterion_bl(pred_baseline, baseline_label)
                    loss_end = self.criterion_end(pred_end, end_label)

                    if loss_bl > 0:
                        # backward + optimize
                        loss_bl.backward()
                        loss_end.backward()
                        self.optimizer.step()

                    running_loss += loss_bl.detach() + loss_end.detach()
                    running_bl_loss += loss_bl.detach()
                    running_end_loss += loss_end.detach()

                    running_counter += 1

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                dbimg = draw_baselines(image=image[0], baselines=pred_list)
                writer.add_image(tag='train/pred', img_tensor=dbimg, global_step=steps)


        loss = running_loss/running_counter
        bl_loss = running_bl_loss/running_counter
        end_loss = running_end_loss/running_counter
        length_loss = running_end_loss/running_counter

        # Write to tensorboard
        writer.add_scalar(tag='mse/train', scalar_value=loss, global_step=steps)
        writer.add_scalar(tag='bl_loss/train', scalar_value=bl_loss, global_step=steps)
        writer.add_scalar(tag='end_loss/train', scalar_value=end_loss, global_step=steps)
        writer.add_scalar(tag='length_loss/train', scalar_value=length_loss, global_step=steps)
        writer.add_scalar(tag='lr', scalar_value=get_lr(self.optimizer), global_step=steps)

        self.scheduler.step()

        return loss, steps

    def _eval(self, eval_steps, writer):
        """
        Validate the model on the validation set. Returns  the loss.
        :return: The loss on the validation set.
        """
        self.model.eval()
        self.model.data_augmentation = False

        tensorboard_img_steps = 499

        running_loss = 0
        running_bl_loss = 0
        running_end_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['eval'], dynamic_ncols=True):
            image = batch['image'].to(self.device)
            baselines = batch['baselines'][0].float().to(self.device)
            bl_lengths = batch['bl_lengths'][0].to(self.device)

            # if self.with_seg is true the model performs a segmentation and uses the segmentation information
            # for computing the baselines.
            if self.with_seg:
                seg_image = batch['seg_image'].to(self.seg_device)
                with torch.no_grad():
                    # seg_out = nn.Softmax(dim=0)(self.seg_model(seg_image)['out']).detach().to(self.device)
                    seg_out = self.seg_model(seg_image)['out'].detach().to(self.device)
                    seg_out = nn.functional.interpolate(seg_out, size=image.size()[2:], mode='nearest')
                    seg_out = nn.Softmax()(seg_out)
                image = torch.cat([image[:, 0:1, :, :], seg_out[:, [self.classes.index('text'), self.classes.index('baselines')], :, :]], dim=1).detach()

                # with torch.no_grad():
                #     seg_out = nn.Softmax()(self.seg_model(image.to(self.seg_device))['out']).detach().to(self.device)
                # image = torch.cat([image[:, 0:1, :, :], seg_out[:, [self.classes.index('end_points'), self.classes.index('baselines')], :, :]], dim=1).detach()

            eval_steps += image.size(0)

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])
            start_points = torch.tensor([[bl[0, 0], bl[0, 1]] for bl in baselines[0:number_of_baselines]])
            end_points = torch.tensor([[bl[-1, 0], bl[-1, 1]] for bl in baselines[0:number_of_baselines]])

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if eval_steps % tensorboard_img_steps == tensorboard_img_steps-1:
                pred_list = []

            with torch.no_grad():
                for n in range(number_of_baselines):

                    # Compute box size
                    box_size = int(get_smallest_distance(start_points[n], start_points))
                    box_size = max(10, min(48, box_size))

                    gt_baseline = baselines[n][:bl_lengths[n]]

                    pred_baseline, pred_end, _ = self.model(img=image, box_size=box_size, baseline=gt_baseline,
                                                            reset_idx=100)

                    # Every tensorboard_img_steps steps save the result to tensorboard:
                    if eval_steps % tensorboard_img_steps == tensorboard_img_steps-1:
                        pred_list.append(pred_baseline)

                    baseline_label, end_label = prepare_data_for_loss(gt_baseline, pred_baseline)
                    baseline_label = baseline_label.to(self.device)
                    end_label = end_label.to(self.device)

                    # The loss is a combination of the regression loss for prediction of the baseline coordinates,
                    # the length of the last baseline segment and the classification loss for the prediction of
                    # the baseline end.
                    loss_bl = self.criterion_bl(pred_baseline, baseline_label)
                    loss_end = self.criterion_end(pred_end, end_label)

                    running_loss += loss_bl.detach() + loss_end.detach()
                    running_bl_loss += loss_bl.detach()
                    running_end_loss += loss_end.detach()

                    running_counter += 1

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if eval_steps % tensorboard_img_steps == tensorboard_img_steps-1:
                dbimg = draw_baselines(image=image[0], baselines=pred_list)
                writer.add_image(tag='eval/pred', img_tensor=dbimg, global_step=eval_steps)

        loss = running_loss/running_counter
        bl_loss = running_bl_loss/running_counter
        end_loss = running_end_loss/running_counter
        length_loss = running_end_loss/running_counter

        # Write to tensorboard
        writer.add_scalar(tag='mse/eval', scalar_value=loss, global_step=eval_steps)
        writer.add_scalar(tag='bl_loss/eval', scalar_value=bl_loss, global_step=eval_steps)
        writer.add_scalar(tag='end_loss/eval', scalar_value=end_loss, global_step=eval_steps)
        writer.add_scalar(tag='length_loss/eval', scalar_value=length_loss, global_step=eval_steps)

        return loss, eval_steps

    def train(self):
        """
        Training routing. Loops over all epochs and calls the training and eval function.
        Prints current stats to the console.
        :return: The trained model
        """

        since = time.time()
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.exp_name))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = float('Inf')

        steps = 0
        eval_steps = 0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            loss, steps = self._train_epoch(steps, writer)
            print('Train: {} loss: {:.4f}'.format('Train', loss))

            # Write to tensorboard
            writer.add_scalar(tag='mse/train', scalar_value=loss,
                              global_step=steps)

            if epoch % self.eval_epoch == 0:
                loss, eval_steps = self._eval(eval_steps, writer)
                print('Eval: {} loss: {:.4f}'.format('Eval', loss))

                # deep copy the model
                if loss <= best_mse:
                    best_mse = loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model, os.path.join(self.output_folder, 'line_rider', self.exp_name + '.pt'))
                    print('## New best! Model saved!')


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval mse: {:4f}'.format(best_mse))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, 'line_rider', self.exp_name + '.pt'))

        writer.close()

        return self.model
