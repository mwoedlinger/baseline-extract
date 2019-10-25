import os
import time
import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from ..data.dataset import BaselineDataset
from ..utils.normalize_baselines import normalize_baselines
from ..model.line_rider import LineRider

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

        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        self.model, self.criterion = self.get_model(weights)

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
        criterion = torch.nn.MSELoss()

        return [model_ft, criterion]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """


        shuffle = {'train': False, 'eval': False}#TODO: change shuffle back to True for train
        batch_size_dict = {'train': self.batch_size, 'eval': 1}

        image_datasets = {inf_type: BaselineDataset(inf_type=inf_type,
                                                    parameters=self.parameters)
                          for inf_type in ['train', 'eval']}

        dataloaders = {inf_type: torch.utils.data.DataLoader(image_datasets[inf_type],
                                                             batch_size=batch_size_dict[inf_type],
                                                             shuffle=shuffle[inf_type],
                                                             num_workers=4)
                       for inf_type in ['train', 'eval']}

        return dataloaders

    def _train_epoch(self, epoch, steps, writer):
        self.model.train()

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):

            image = batch['image']
            baselines = batch['baselines'][0]
            start_points = batch['start_points'][0]
            start_angles = batch['start_angles'][0]
            bl_lengths = batch['bl_lengths'][0]

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1 ])

            image = image.to(self.device)
            start_points = start_points.to(self.device)
            start_angles = start_angles.to(self.device)

            box_size = 30 #TODO: compute box_size

            steps += image.size(0)

            with torch.set_grad_enabled(True):
                for n in range(number_of_baselines):

                    # zero the parameter gradients
                    self.optimizer.zero_grad()


                    # TODO: implement data augmentation: vary the baseline a little bit

                    bl = baselines[n][:bl_lengths[n]]
                    bl_n = normalize_baselines(bl, box_size, device=self.device)
                    bl_n = bl_n.to(self.device)

                    spoint = start_points[n]
                    sangle = start_angles[n]

                    x_list, y_list = self.model(img=image, x_0=spoint[0], y_0=spoint[1], angle=sangle,
                                                box_size=box_size)

                    # Match the length. In case the network didn't find the ending properly we add the last predicted
                    # point repeatedly to match the dimension of bl_n
                    if len(x_list) < len(bl_n):
                        diff = len(bl_n) - len(x_list)
                        x_list = torch.cat([x_list, torch.tensor([x_list[-1]] * diff)], dim=0)
                        y_list = torch.cat([y_list, torch.tensor([y_list[-1]] * diff)], dim=0)
                    elif len(x_list) > len(bl_n):
                        diff = len(x_list) - len(bl_n)
                        bl_n = torch.cat([bl_n, torch.stack([bl_n[-1]] * diff)], dim=0)

                    # Now combine the tensors in these lists to a single tensor and and permute the indices
                    # such that the two tensors match.
                    l_target = torch.cat([l.unsqueeze(0) for l in bl_n], dim=0).float().to(self.device)
                    l_input = torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0).to(self.device)


                    loss = self.criterion(l_input, l_target)
                    loss.requires_grad = True
                    # loss = self.criterion(out, bl_n[-1])
                    # print('## loss: ' + str(loss.requires_grad))

                    # backward + optimize
                    #print(x_list[0].requires_grad)
                    loss.backward()
                    for param in self.model.parameters():
                        print(param.grad)
                    self.optimizer.step()

                    # Write to tensorboard
                    writer.add_scalar(tag='loss/train', scalar_value=loss,
                                      global_step=steps)

                    #TODO: compute accuracy
                    acc = 0.0
                    #TODO: test if it is better to compute the loss for the whole document instead of single baselines.


        return loss, acc, steps

    def _eval(self, writer):
        self.model.eval()

        # Iterate over data.
        for batch in tqdm(self.dataloaders['eval']):
            image = batch['image']
            baselines = batch['baselines'][0]
            start_points = batch['start_points'][0]
            start_angles = batch['start_angles'][0]
            bl_lengths = batch['bl_lengths'][0]

            # Compute the number of baselines. The baselines are padded with [-1] and [-1,-1].
            # number_of_baselines is the real number of baselines not counting the padding and bl_lengths[idx]
            # is the real (not counting the padding) length of baseline idx.
            number_of_baselines = min([idx for idx in range(0, len(bl_lengths)) if bl_lengths[idx] == -1 ])

            image = image.to(self.device)
            start_points = start_points.to(self.device)
            start_angles = start_angles.to(self.device)

            box_size = 128 #TODO: compute box_size

            for n in range(number_of_baselines):
                bl = baselines[n][:bl_lengths[n]]
                bl_n = normalize_baselines(bl, box_size, device=self.device)
                bl_n = bl_n.to(self.device)

                spoint = start_points[n]
                sangle = start_angles[n]

                x_list, y_list = self.model(img=image, x_0=spoint[0], y_0=spoint[1], angle=sangle,
                                            box_size=box_size)

                # Match the length. In case the network didn't find the ending properly we add the last predicted
                # point repeatedly to match the dimension of bl_n
                if len(x_list) < len(bl_n):
                    diff = len(bl_n) - len(x_list)
                    x_list = torch.cat([x_list, torch.tensor([x_list[-1]] * diff)], dim=0)
                    y_list = torch.cat([y_list, torch.tensor([y_list[-1]] * diff)], dim=0)
                elif len(x_list) > len(bl_n):
                    diff = len(x_list) - len(bl_n)
                    bl_n = torch.cat([bl_n, torch.stack([bl_n[-1]] * diff)], dim=0)

                # Now combine the tensors in these lists to a single tensor and and permute the indices
                # such that the two tensors match.
                l_target = torch.cat([l.unsqueeze(0) for l in bl_n], dim=0).float().to(self.device)
                l_input = torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0).to(self.device)

                loss = self.criterion(l_input, l_target)

                # TODO: compute accuracy
                acc = 0.0
                # TODO: test if it is better to compute the loss for the whole document instead of single baselines.

        return loss, acc


    def train(self):
        since = time.time()
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.exp_name))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        steps = 0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            loss, acc, steps = self._train_epoch(epoch, steps, writer)
            print('{} loss: {:.4f} acc: {:.4f}'.format('Train', loss, acc))

            if epoch > 1 and epoch % self.eval_epoch == 0:
                loss, acc = self._eval(writer)
                print('{} loss: {:.4f} acc: {:.4f}'.format('Eval', loss, acc))

                # Write to tensorboard
                writer.add_scalar(tag='loss/eval', scalar_value=loss,
                                  global_step=steps)

                # deep copy the model
                if acc >= best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model, os.path.join(self.output_folder, self.exp_name + '.pt'))


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, self.exp_name + '.pt'))

        writer.close()

        return self.model
