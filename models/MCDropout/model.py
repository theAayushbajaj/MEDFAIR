import torch
import torch.nn as nn
from models.basenet import BaseNet
from models.utils import standard_val, standard_test, standard_train

class MCDropout(BaseNet):
    def __init__(self, opt, wandb):
        super(MCDropout, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
        self.num_samples = opt['num_samples']
        self.temperature = opt['temperature']

    def set_network(self, opt):
        """Define the network"""
        # TODO
        pass

    def _train(self, loader):
        """Train the model for one epoch"""
        self.network.train()
        auc, train_loss, ece = standard_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb)

        print('Training epoch {}: AUC:{} ECE:{}'.format(self.epoch, auc, ece))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))

        self.epoch += 1

    def _val(self, loader):
        """Validate the model"""
        self.network.eval()
        auc, val_loss, ece = standard_val(self.opt, self.network, loader, self._criterion, self.wandb)

        print('Validation epoch {}: AUC:{} ECE:{}'.format(self.epoch, auc, ece))
        print('Validation epoch {}: loss:{}'.format(self.epoch, val_loss))

        return auc, val_loss, ece

    def _test(self, loader):
        """Test the model"""
        self.network.eval()
        auc, test_loss, ece = standard_test(self.opt, self.network, loader, self._criterion, self.wandb)

        print('Test epoch {}: AUC:{} ECE:{}'.format(self.epoch, auc, ece))
        print



