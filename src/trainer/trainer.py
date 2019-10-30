import os
import time
import math
import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from ..data.dataset_line_rider import DatasetLineRider
from ..utils.normalize_baselines import normalize_baselines
from ..utils.visualization import draw_baselines
from ..model.line_rider import LineRider
from ..model.loss import L12Loss


class Trainer:
    """
    Performs the training of the LineRider network.
    """

    def __init__(self, config, weights=None):
        self.exp_name = config['exp_name']
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
        # criterion_bl = torch.nn.MSELoss()
        # criterion_bl = torch.nn.L1Loss()
        criterion_end = torch.nn.NLLLoss()
        criterion_bl = L12Loss()

        return [model_ft, criterion_bl, criterion_end]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': False, 'eval': False}#TODO: change shuffle back to True for train
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
        # TODO: o) Compute box_size
        #       o) Implement data augmentation: vary the baseline a little bit
        #       o) Test if it is better to compute the loss for the whole document instead of single baselines.
        self.model.train()

        if epoch % 4 == 0 and epoch > 0:
            if self.reset_idx < 4:
                self.reset_idx += 1

        tensorboard_img_steps = 150

        running_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):

            image = batch['image']
            baselines = batch['baselines'][0]
            bl_lengths = batch['bl_lengths'][0]

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1])

            image = image.to(self.device)

            box_size = 128

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
                    bl_n = normalize_baselines(bl, box_size, device=self.device)
                    bl_n = bl_n.to(self.device)

                    c_list, bl_end_list = self.model(img=image, box_size=box_size, baseline=bl_n,
                                                     reset_idx=self.reset_idx)

                    # Every tensorboard_img_steps steps save the result to tensorboard:
                    if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                        pred_list.append(c_list)

                    # Match the length. In case the network didn't find the ending properly we add the last predicted
                    # point repeatedly to match the dimension of bl_n
                    if len(c_list) < len(bl_n):
                        diff = len(bl_n) - len(c_list)
                        c_list = torch.cat([c_list, torch.stack([c_list[-1]] * diff)], dim=0)
                    elif len(c_list) > len(bl_n):
                        diff = len(c_list) - len(bl_n)
                        bl_n = torch.cat([bl_n, torch.stack([bl_n[-1]] * diff)], dim=0)

                    l_label = bl_n
                    l_pred = c_list
                    # l_bl_end_label = torch.tensor([0]*(len(bl_n)-1)+[1]).float().to(self.device)
                    l_bl_end_label = torch.tensor([len(bl_n)-1]).to(self.device)
                    if len(bl_n) > len(bl_end_list):
                        l_bl_end_pred = torch.cat([bl_end_list,
                                                   torch.tensor([0]*(len(bl_n)-len(bl_end_list))).float().to(self.device)], dim=0)
                    else:
                        l_bl_end_pred = bl_end_list
                    l_bl_end_pred = l_bl_end_pred.unsqueeze(0)

                    # The loss is a combination of the regression loss for the baseline prediction and the
                    # classification for the prediction of the baseline end.
                    loss = self.criterion_bl(l_pred, l_label)*(2+self.criterion_end(l_bl_end_pred, l_bl_end_label))

                    running_loss += loss
                    running_counter += 1

                    # backward + optimize
                    loss.backward()
                    # print('## in:  ' + str(l_label))
                    # print('## out: ' + str(l_pred))
                    # print('## diff: ' + str(l_pred - l_label))
                    # print('## bl_end: ' + str(bl_end_list))
                    self.optimizer.step()

                    # Write to tensorboard
                    writer.add_scalar(tag='mse/train', scalar_value=loss,
                                      global_step=steps)

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                dbimg = draw_baselines(image=image[0], baselines=pred_list, bl_lengths=bl_lengths)
                writer.add_image(tag='train/pred', img_tensor=dbimg, global_step=steps)

        loss = running_loss/running_counter

        return loss, steps

    def _eval(self, writer):
        # TODO: compute box_size
        self.model.eval()

        running_loss = 0
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

            image = image.to(self.device)

            box_size = 64

            for n in range(number_of_baselines):
                # Normalise the baselines such that each line segment has the length 'box_size'
                bl = baselines[n][:bl_lengths[n]]
                bl_n = normalize_baselines(bl, box_size, device=self.device)
                bl_n = bl_n.to(self.device)

                x_0 = bl_n[0, 0]
                y_0 = bl_n[0, 1]
                if torch.abs(bl_n[0, 0] - bl_n[0 + 1, 0]) < 0.001:
                    angle = torch.tensor(math.pi / 2.0).to(self.device)
                else:
                    angle = torch.atan(
                        (bl_n[0, 1] - bl_n[0 + 1, 1]) / (bl_n[0, 0] - bl_n[0 + 1, 0]))

                c_list, bl_end_list = self.model(img=image, box_size=box_size, x_0=bl_n[0, 0], y_0=bl_n[0, 1],
                                                 angle_0=angle, reset_idx=1000)

                # Match the length. In case the network didn't find the ending properly we add the last predicted
                # point repeatedly to match the dimension of bl_n
                if len(c_list) < len(bl_n):
                    diff = len(bl_n) - len(c_list)
                    c_list = torch.cat([c_list, torch.stack([c_list[-1]] * diff)], dim=0)
                elif len(c_list) > len(bl_n):
                    diff = len(c_list) - len(bl_n)
                    bl_n = torch.cat([bl_n, torch.stack([bl_n[-1]] * diff)], dim=0)

                l_label = bl_n
                l_pred = c_list
                l_bl_end_label = torch.tensor([0] * (len(l_pred) - 1) + [1]).float().to(self.device)
                l_bl_end_pred = bl_end_list

                loss = self.criterion_bl(l_pred, l_label)  # + self.criterion_end(l_bl_end_pred, l_bl_end_label)

                running_loss += loss
                running_counter += 1

        loss = running_loss/running_counter

        return loss


    def train(self):
        since = time.time()
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.exp_name))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = float('Inf')
        steps = 0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            loss, steps = self._train_epoch(steps, epoch, writer)
            print('{} loss: {:.4f}'.format('Train', loss))

            if epoch > 1 and epoch % self.eval_epoch == 0:
                loss = self._eval(writer)
                print('{} loss: {:.4f}'.format('Eval', loss))

                # Write to tensorboard
                writer.add_scalar(tag='mse/eval', scalar_value=loss,
                                  global_step=steps)

                # deep copy the model
                if loss <= best_mse:
                    best_mse = loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model, os.path.join(self.output_folder, self.exp_name + '.pt'))


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval mse: {:4f}'.format(best_mse))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, self.exp_name + '.pt'))

        writer.close()

        return self.model
