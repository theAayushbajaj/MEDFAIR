import torch
import torch.nn as nn
from models.basenet import BaseNet
from models.utils import standard_val, standard_test, standard_train

from resnet50_mcdropout import cusResNet50_mcdropout

class MCDropout(BaseNet):
    def __init__(self, opt, wandb):
        super(MCDropout, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
        self.num_samples = opt['num_samples']

    def set_network(self, opt):
        """Define the network"""
        self.network = cusResNet50_mcdropout(dropout_rate=opt['dropout_rate'], num_classes=self.output_dim)

    def _train(self, loader):
        """Train the model for one epoch"""
        self.network.train()
        auc, train_loss, ece = standard_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb)

        print('Training epoch {}: AUC:{} ECE:{}'.format(self.epoch, auc, ece))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))

        self.epoch += 1



