import os
import time
import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from ..data.dataset_line_finder import DatasetLineFinder
from ..utils.visualization import draw_start_points
from ..model.line_finder import LineFinder
from ..model.loss import LineFinderLoss
from ..segmentation.gcn_model import GCN


class TrainerLineFinder:
    """
    Performs the training of the LineFinder network.
    """

    def __init__(self, config, weights=None):
        self.exp_name = 'line_finder_' + config['exp_name']
        self.log_dir = config['log_dir']
        self.output_folder = config['output_folder']

        self.lr = config['lr']
        self.gpu = config['gpu']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.eval_epoch = config['eval_epoch']
        self.parameters = config['data']
        self.max_side = config['data']['max_side']
        self.reset_idx = config['reset_idx_start']

        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        self.model, self.criterion = self.get_model(weights)

        self.optimizer, self.scheduler = self.get_optimizer()
        self.dataloaders = self.get_dataloaders()

        self.segmentation_weights = config['segmentation']['segmentation_weights']
        if self.segmentation_weights is not None:
            self.segmentation_gpu = config['segmentation']['segmentation_gpu']
            self.segmentation_device = torch.device('cuda:' + str(self.segmentation_gpu) if torch.cuda.is_available()
                                                    else 'cpu')

            print('## Loading segmentation model')
            self.seg_model = GCN(n_classes=config['segmentation']['segmentation_classes'])
            self.seg_model.load_state_dict(torch.load(self.segmentation_weights, map_location=self.segmentation_device))
            self.seg_model.to(self.segmentation_device)
            self.seg_model.eval()

            self.segmentation = True
        else:
            self.segmentation = False

    def get_model(self, weights):
        """
        Load the correct model,
        :return: a list containing the model and the loss function (CrossEntropyLoss)
        """

        if not weights:
            model_ft = LineFinder(device=self.device)
        else:
            model_ft = torch.load(weights, map_location=self.device)

        model_ft.to(self.device)
        criterion = LineFinderLoss()

        return [model_ft, criterion]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)#20, 0.5#30

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': True, 'eval': False}
        batch_size_dict = {'train': self.batch_size, 'eval': 1}

        image_datasets = {inf_type: DatasetLineFinder(inf_type=inf_type,
                                                      parameters=self.parameters,
                                                      random_seed=42)
                          for inf_type in ['train', 'eval']}

        dataloaders = {inf_type: torch.utils.data.DataLoader(image_datasets[inf_type],
                                                             batch_size=batch_size_dict[inf_type],
                                                             shuffle=shuffle[inf_type],
                                                             num_workers=4)
                       for inf_type in ['train', 'eval']}

        return dataloaders

    def _train_epoch(self, steps, writer):
        """
        Train the model for one epoch. returns a a tuple containing the loss and the current step counter.
        :param steps: starting point for step counter. The function returns the updated steps.
        :param writer: Tensorboard writer
        :return: A tuple containing the loss and the current step counter.
        """
        self.model.train()

        tensorboard_img_steps = 47

        running_loss = 0
        running_loc_loss = 0
        running_conf_loss = 0
        running_conf_anti_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):
            image = batch['image'].to(self.device)
            label = batch['label'].to(self.device)
            label_len = batch['label_len'].to(self.device)

            if self.segmentation:
                with torch.no_grad():
                    seg_out = self.seg_model(image)[0]
                    seg_out.detach()
                image = (image[:, 0:1, :, :] + image[:, 1:2, :, :] + image[:, 2:3, :, :])/3.0
                image = torch.cat([image, seg_out[:, [0, 2], :, :]], dim=1).detach()

            steps += 1#image.size(0)

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                out = self.model(image)
                loss, loc_loss, conf_loss, conf_anti_loss = self.criterion(out, label, label_len)

                running_loss += loss
                running_loc_loss += loc_loss
                running_conf_loss += conf_loss
                running_conf_anti_loss += conf_anti_loss
                running_counter += image.size(0)

                # backward + optimize
                loss.backward()
                self.optimizer.step()

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                combined = draw_start_points(image=image[0].cpu(), label=out[0].detach(), true_label=label[0][0:label_len[0]])
                writer.add_image(tag='train/pred', img_tensor=combined, global_step=steps)


        loss = running_loss/running_counter
        loc_loss = running_loc_loss/running_counter
        conf_loss = running_conf_loss/running_counter
        conf_anti_loss = running_conf_anti_loss/running_counter

        # Write to tensorboard
        writer.add_scalar(tag='loss/train', scalar_value=loss, global_step=steps)
        writer.add_scalar(tag='loc_loss/train', scalar_value=loc_loss, global_step=steps)
        writer.add_scalar(tag='conf_loss/train', scalar_value=conf_loss, global_step=steps)
        writer.add_scalar(tag='conf_anti_loss/train', scalar_value=conf_anti_loss, global_step=steps)

        self.scheduler.step()

        return loss, loc_loss, conf_loss, conf_anti_loss, steps

    def _eval(self, writer, eval_steps):
        self.model.eval()

        tensorboard_img_steps = 51

        running_loss = 0
        running_loc_loss = 0
        running_conf_loss = 0
        running_counter = 0

        # Iterate over data.
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['eval']):

                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                label_len = batch['label_len'].to(self.device)

                if self.segmentation:
                    with torch.no_grad():
                        seg_out = self.seg_model(image)[0]
                        seg_out.detach()
                    image = (image[:, 0:1, :, :] + image[:, 1:2, :, :] + image[:, 2:3, :, :]) / 3.0
                    image = torch.cat([image, seg_out[:, [0, 2], :, :]], dim=1).detach()

                eval_steps += 1#image.size(0)

                out = self.model(image)
                loss, loc_loss, conf_loss, conf_anti_loss = self.criterion(out, label, label_len)
                conf_loss = conf_loss + conf_anti_loss

                running_loss += loss
                running_loc_loss += loc_loss
                running_conf_loss += conf_loss
                running_counter += image.size(0)

                # Every tensorboard_img_steps steps save the result to tensorboard:
                if eval_steps % tensorboard_img_steps == tensorboard_img_steps - 1:
                    combined = draw_start_points(image=image[0].cpu(), label=out[0].detach(), true_label=label[0][0:label_len[0]])
                    writer.add_image(tag='eval/pred', img_tensor=combined, global_step=eval_steps)

        loss = running_loss/running_counter
        loc_loss = running_loc_loss/running_counter
        conf_loss = running_conf_loss/running_counter

        # Write to tensorboard
        writer.add_scalar(tag='loss/eval', scalar_value=loss, global_step=eval_steps)
        writer.add_scalar(tag='loc_loss/eval', scalar_value=loc_loss, global_step=eval_steps)
        writer.add_scalar(tag='conf_loss/eval', scalar_value=conf_loss, global_step=eval_steps)

        return loss, loc_loss, conf_loss, eval_steps


    def train(self):
        since = time.time()
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.exp_name))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = float('Inf')
        steps = 0
        eval_steps = 0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            loss, loc_loss, conf_loss, conf_anti_loss, steps = self._train_epoch(steps, writer)
            print('{}:\nLoss: {:.4f} \nLoc_loss: {:.4f} \nConf_loss: {:.4f} \nConf_anti_loss: {:4f}'.format(
                'Train', loss, loc_loss, conf_loss, conf_anti_loss))

            if epoch > 1 and epoch % self.eval_epoch == 0:
                loss, loc_loss, conf_loss, eval_steps = self._eval(writer, eval_steps)
                print('{} loss: {:.4f}'.format('Eval', loss))

                # Write to tensorboard
                writer.add_scalar(tag='loss/eval', scalar_value=loss, global_step=eval_steps)
                writer.add_scalar(tag='loc_loss/eval', scalar_value=loc_loss, global_step=eval_steps)
                writer.add_scalar(tag='conf_loss/eval', scalar_value=conf_loss, global_step=eval_steps)

                # deep copy the model
                if loss <= best_mse:
                    best_mse = loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model, os.path.join(self.output_folder, 'line_finder', self.exp_name + '.pt'))


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval mse: {:4f}'.format(best_mse))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, 'line_finder',  self.exp_name + '.pt'))

        writer.close()

        return self.model
