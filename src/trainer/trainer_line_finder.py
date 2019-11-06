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


class TrainerLineFinder:
    """
    Performs the training of the LineFinder network.
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

        self.model, self.criterion = self.get_model(weights)

        self.optimizer, self.scheduler = self.get_optimizer()
        self.dataloaders = self.get_dataloaders()

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': False, 'eval': False}#TODO: change to True for train
        batch_size_dict = {'train': self.batch_size, 'eval': 1}

        image_datasets = {inf_type: DatasetLineFinder(inf_type=inf_type,
                                                      parameters=self.parameters)
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

        tensorboard_img_steps = 99

        running_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):

            image = batch['image'].to(self.device)
            label = batch['label'].to(self.device)

            steps += image.size(0)

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                t1 = time.time()
                out = self.model(image)
                t2 = time.time()
                print('model: dt = ' + str(t2-t1))

                t1 = time.time()
                loss = self.criterion(out, label)
                t2 = time.time()
                print('loss:  dt = ' + str(t2-t1))

                running_loss += loss
                running_counter += 1

                # backward + optimize
                loss.backward()
                self.optimizer.step()

                # Write to tensorboard
                writer.add_scalar(tag='mse/train', scalar_value=loss,
                                  global_step=steps)


            # Every tensorboard_img_steps steps save the result to tensorboard:
            if steps % tensorboard_img_steps == tensorboard_img_steps-1:
                combined = draw_start_points(image=image, label=label)
                writer.add_image(tag='train/pred', img_tensor=combined, global_step=steps)

        loss = running_loss/running_counter

        return loss, steps

    def _eval(self, writer, eval_steps):
        self.model.eval()

        tensorboard_img_steps = 99

        running_loss = 0
        running_counter = 0

        # Iterate over data.
        for batch in tqdm(self.dataloaders['train']):

            image = batch['image'].to(self.device)
            label = batch['label'].to(self.device)

            eval_steps += image.size(0)

            out = self.model(image)
            loss = self.criterion(out, label)

            running_loss += loss
            running_counter += 1

            # Every tensorboard_img_steps steps save the result to tensorboard:
            if eval_steps % tensorboard_img_steps == tensorboard_img_steps - 1:
                combined = draw_start_points(image=image, label=label)
                writer.add_image(tag='eval/pred', img_tensor=combined, global_step=steps)

        loss = running_loss / running_counter

        return loss, eval_steps


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

            loss, steps = self._train_epoch(steps, writer)
            print('Train: {} loss: {:.4f}'.format('Train', loss))

            if epoch > 1 and epoch % self.eval_epoch == 0:
                loss, eval_steps = self._eval(writer, eval_steps)
                print('Eval: {} loss: {:.4f}'.format('Eval', loss))

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
