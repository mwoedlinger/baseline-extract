import os
import time
import copy
import tqdm
import torch
from ..data.dataset import BaselineDataset

class Trainer:
    """
    Performs the training of the LineRider network.
    """

    def __init__(self, model, config, weights=None):
        self.exp_name = config['exp_name']
        self.log_dir = config['log_dir']
        self.output_folder = config['output_folder']

        self.lr = config['lr']
        self.gpu = config['gpu']
        self.batch_size = 1#config['batch_size']
        self.epochs = config['epochs']
        self.eval_epoch = config['eval_epoch']
        self.parameters = config['data']

        self.device = torch.device('cuda:' + self.cfg_gpu if torch.cuda.is_available() else 'cpu')

        self.model, self.criterion = self.get_model(model, weights)

        self.optimizer, self.scheduler = self.get_optimizer()
        self.dataloaders = self.get_dataloaders()

    def get_model(self, model, weights):
        """
        Load the correct model,
        :return: a list containing the model and the loss function (CrossEntropyLoss)
        """

        if not weights:
            raise  NotImplementedError
        else:
            model_ft = torch.load(weights, map_location=self.device)

        #TODO: write proper criterion. The baselines consist of pairs of number not just a single list of numbers.
        criterion = torch.nn.MSELoss()

        return [model_ft, criterion]

    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        return [optimizer, exp_lr_scheduler]

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        There is currently no data augmentation implemented.
        :return: The dataloaders
        """

        shuffle = {'train': True, 'eval': False}
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

            image = batch['image'][0]
            baselines = batch['baselines'][0]
            start_points = batch['start_points'][0]
            start_angles = batch['start_angles'][0]

            image = image.to(self.device)
            baselines = baselines.to(self.device)
            start_points = start_points.to(self.device)
            start_angles = start_angles.to(self.device)

            #TODO: compute scale
            #TODO: Normalize baselines, Need scale for that

            steps += image.size(0)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                for n in range(len(baselines)):
                    bl = baselines[n]
                    spoint = start_points[n]
                    sangle = start_angles[n]
                    outputs = self.model(img=image, x_0=spoint[0], y_0=spoint[1], angle=sangle, scale=scale)

                    loss = self.criterion(outputs, bl)

                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()

                    #TODO: compute accuracy
                    #TODO: test if it is better to compute the loss for the whole document instead of single baselines.

        return loss, acc

    def _eval(self, writer):
        self.mode.eval()
        raise NotImplementedError

        return loss, acc


    def train(self):
        since = time.time()
        writer = torch.utils.tensorboard.SummaryWriter(logdir=os.path.join(self.log_dir, self.exp_name))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        steps = 0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            loss, acc = self._train_epoch(epoch, steps, writer)
            print('{} loss: {:.4f} acc: {:.4f}'.format('Train', loss, acc))

            if epoch % self.eval_epoch == 0:
                loss, acc = self._eval(writer)
                print('{} loss: {:.4f} acc: {:.4f}'.format('Eval', loss, acc))

                # deep copy the model
                if acc >= best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.save_model()


        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, os.path.join(self.output_folder, self.exp_name + '.pt'))

        writer.close()

        return self.model
