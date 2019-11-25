import os
import time
import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from ..data.dataset_line_rider import DatasetLineRider, prepare_data_for_loss
from ..utils.normalize_baselines import normalize_baselines, compute_start_and_angle
from ..utils.visualization import draw_baselines
from ..model.line_rider import LineRider
from ..model.loss import L12Loss
from ..utils.distances import d2
from ..utils.distances import get_median_diff


class TrainerLineRider:
    """
    Performs the training of the LineRider network.
    """

    def __init__(self, config, weights=None):
        self.exp_name = 'line_rider_' + config['exp_name']
        self.log_dir = config['log_dir']
        self.output_folder = config['output_folder']

        self.lr = config['lr']
        self.gpu = config['gpu']
        self.batch_size = 1#config['batch_size']
        self.epochs = config['epochs']
        self.eval_epoch = config['eval_epoch']
        self.parameters = config['data']
        self.max_side = config['data']['max_side']
        self.reset_idx = config['reset_idx_start']

        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        self.model, self.criterion_bl, self.criterion_end = self.get_model(weights)

        self.optimizer, self.scheduler = self.get_optimizer()
        self.dataloaders = self.get_dataloaders()

    def get_model(self, weights):
        """
        Load the correct model,
        :return: a list containing the model and the loss function (CrossEntropyLoss)
        """

        if not weights:
            model_ft = LineRider(device=self.device)
        else:
            model_ft = torch.load(weights, map_location=self.device)

        model_ft.to(self.device)
        criterion_bl = torch.nn.MSELoss()
        # criterion_bl = torch.nn.L1Loss()
        criterion_end = torch.nn.BCELoss()
        # criterion_bl = L12Loss()
        self.criterion_length = torch.nn.MSELoss()

        return [model_ft, criterion_bl, criterion_end]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': True, 'eval': False}
        batch_size_dict = {'train': self.batch_size, 'eval': 1}

        image_datasets = {inf_type: DatasetLineRider(inf_type=inf_type,
                                                    parameters=self.parameters)
                          for inf_type in ['train', 'eval']}

        dataloaders = {inf_type: torch.utils.data.DataLoader(image_datasets[inf_type],
                                                             batch_size=batch_size_dict[inf_type],
                                                             shuffle=shuffle[inf_type],
                                                             num_workers=4)
                       for inf_type in ['train', 'eval']}

        return dataloaders

    def _train_epoch(self, steps, epoch, writer):
        """
        Train the model for one epoch. Returns a a tuple containing the loss and the current step counter.
        :param steps: starting point for step counter. The function returns the updated steps.
        :param epoch: the current epoch number
        :param writer: Tensorboard writer
        :return: A tuple containing the loss and the current step counter.
        """
        # TODO: x) Compute box_size
        #       x) Implement data augmentation: vary the baseline a little bit
        #       o) Test if it is better to compute the loss for the whole document instead of single baselines.
        self.model.train()

        if self.reset_idx < 8:
            self.reset_idx += 1
        # if epoch % 1 == 0 and epoch > 0:
        #     self.reset_idx += 1

        tensorboard_img_steps = 499

        running_loss = 0
        running_bl_loss = 0
        running_end_loss = 0
        running_length_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):
            image = batch['image'].to(self.device)
            baselines = batch['baselines'][0]
            bl_lengths = batch['bl_lengths'][0]

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])

            start_points = torch.tensor([[bl[0, 0], bl[0, 1]] for bl in baselines[0:number_of_baselines]])
            box_size = get_median_diff(start_points)/2.0
            box_size = min(10, max(32, box_size))

            steps += image.size(0)

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                pred_list = []

            with torch.set_grad_enabled(True):
                for n in range(number_of_baselines):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Normalise the baselines such that each line segment has the length 'box_size'
                    bl = baselines[n][:bl_lengths[n]]
                    bl_n = normalize_baselines(bl, 2*box_size)
                    bl_n = bl_n.to(self.device)
                    bl_n_end_length = d2(bl_n[-1, 0], bl_n[-1, 1], bl_n[-2, 0], bl_n[-2, 1])/(2*box_size)

                    c_list, bl_end_list, bl_end_length_list, _ = self.model(img=image, box_size=box_size, baseline=bl_n,
                                                                         reset_idx=self.reset_idx)

                    # Every tensorboard_img_steps steps save the result to tensorboard:
                    if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                        pred_list.append(c_list)
                        print(str(n) + ': ' + str(c_list))

                    l_label, l_pred, l_bl_end_label, l_bl_end_pred, l_bl_end_length_label, l_bl_end_length_pred = \
                        prepare_data_for_loss(bl_n, c_list, bl_end_list, bl_n_end_length,
                                              bl_end_length_list, box_size, self.device)

                    # The loss is a combination of the regression loss for prediction of the baseline coordinates,
                    # the length of the last baseline segment and the classification loss for the prediction of
                    # the baseline end.
                    loss = self.criterion_bl(l_pred, l_label) \
                           + 10*self.criterion_end(l_bl_end_pred, l_bl_end_label) \
                           + 10*self.criterion_length(l_bl_end_length_pred, l_bl_end_length_label)

                    running_loss += loss
                    running_bl_loss += self.criterion_bl(l_pred, l_label)
                    running_end_loss += 10 * self.criterion_end(l_bl_end_pred, l_bl_end_label)
                    running_length_loss += 10 * self.criterion_length(l_bl_end_length_pred, l_bl_end_length_label)

                    running_counter += 1


                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()

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


        return loss, steps

    def _eval(self, eval_steps, writer):
        """
        Validate the model on the validation set. Returns  the loss.
        :return: The loss on the validation set.
        """
        # TODO: compute box_size
        self.model.eval()

        tensorboard_img_steps = 499

        running_loss = 0
        running_bl_loss = 0
        running_end_loss = 0
        running_length_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['eval']):
            image = batch['image']
            baselines = batch['baselines'][0]
            bl_lengths = batch['bl_lengths'][0]

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])

            start_points = torch.tensor([[bl[0, 0], bl[0, 1]] for bl in baselines[0:number_of_baselines]])
            box_size = get_median_diff(start_points)/2.0
            box_size = min(10, max(32, box_size))

            image = image.to(self.device)

            eval_steps += image.size(0)

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if eval_steps % tensorboard_img_steps == tensorboard_img_steps-1:
                pred_list = []

            with torch.no_grad():
                for n in range(number_of_baselines):
                    # Normalise the baselines such that each line segment has the length 'box_size'
                    bl = baselines[n][:bl_lengths[n]]
                    bl_n = normalize_baselines(bl, 2*box_size) #TODO: make it clearer that 2*box_size is used
                    bl_n = bl_n.to(self.device)
                    bl_n_end_length = d2(bl_n[-1, 0], bl_n[-1, 1], bl_n[-2, 0], bl_n[-2, 1])/(2*box_size)

                    x_0, y_0, angle = compute_start_and_angle(baseline=bl_n, idx=0)
                    x_0 = x_0.to(self.device)
                    y_0 = y_0.to(self.device)
                    angle = angle.to(self.device)

                    # c_list, bl_end_list, bl_end_length_list, _ = self.model(img=image, box_size=box_size, x_0=x_0, y_0=y_0,
                    #                                                      angle_0=angle, reset_idx=1000) TODO: CHANGE BACK!!!!!!!!!
                    c_list, bl_end_list, bl_end_length_list, _ = self.model(img=image, box_size=box_size, baseline=bl_n,
                                                                         reset_idx=30)

                    # Every tensorboard_img_steps steps save the result to tensorboard:
                    if eval_steps % tensorboard_img_steps == tensorboard_img_steps-1:
                        pred_list.append(c_list)

                    l_label, l_pred, l_bl_end_label, l_bl_end_pred, l_bl_end_length_label, l_bl_end_length_pred = \
                        prepare_data_for_loss(bl_n, c_list, bl_end_list, bl_n_end_length,
                                              bl_end_length_list, box_size, self.device)

                    # The loss is a combination of the regression loss for prediction of the baselinecoordinates,
                    # the length of the last baseline segment and the classification loss for the prediction of
                    # the baseline end.
                    loss = self.criterion_bl(l_pred, l_label) \
                           + 10 * self.criterion_end(l_bl_end_pred, l_bl_end_label) \
                           + 10 * self.criterion_length(l_bl_end_length_pred, l_bl_end_length_label)

                    running_loss += loss
                    running_bl_loss += self.criterion_bl(l_pred, l_label)
                    running_end_loss += 10 * self.criterion_end(l_bl_end_pred, l_bl_end_label)
                    running_length_loss += 10 * self.criterion_length(l_bl_end_length_pred, l_bl_end_length_label)

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

            loss, steps = self._train_epoch(steps, epoch, writer)
            print('Train: {} loss: {:.4f}'.format('Train', loss))

            # Write to tensorboard
            writer.add_scalar(tag='mse/train', scalar_value=loss,
                              global_step=steps)

            if epoch > 1 and epoch % self.eval_epoch == 0:
                loss, eval_steps = self._eval(eval_steps, writer)
                print('Eval: {} loss: {:.4f}'.format('Eval', loss))

                # deep copy the model
                if loss <= best_mse:
                    best_mse = loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model, os.path.join(self.output_folder, 'line_rider', self.exp_name + '.pt'))


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval mse: {:4f}'.format(best_mse))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, 'line_rider', self.exp_name + '.pt'))

        writer.close()

        return self.model