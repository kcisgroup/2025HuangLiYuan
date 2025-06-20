import copy
import math

import torch
import torch.optim as optim
import torch.nn as nn
from .wrapper import dmgan_class
from ..utils.train_eval import train, evaluate, train_dp
from opacus import PrivacyEngine
from ..protection.UDP import add_dp_noise
from ..utils.getsize import get_ordereddict_size
from ..utils.lowrank_decomposition import low_rank_decomposition_svd,cnn_prune_terngrad
import numpy as np
from collections import OrderedDict


class Client:

    def __init__(
        self,
        client_id=None,
        client_group=None,
        client_model=None,
        local_epochs=1,
        lr=0.1,
        lr_decay=0.95,
        momentum=0.5,
        init_ratio=0.8,
        global_params=None,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        device=None,
        compensate=None
    ):
        self.client_id = client_id
        self.client_group = client_group
        self.device = device
        self.cur_round = 0
        self.init_ratio = init_ratio
        self.client_model = client_model.to(self.device)
        self.num_epochs = local_epochs
        self.optimizer = optim.SGD(self.client_model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)
        self.lr_decay = lr_decay    # lr decay for each FL round
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.global_params = global_params
        self.compensate = compensate

    def reinit(self):
        for m in self.client_model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


    def synchronize(self, cur_round, model_params):
        self.cur_round = cur_round
        # inner deep copied
        # if cur_round == 0:
        #     self.reinit()
        # else:
        self.global_params = copy.deepcopy(model_params)
        self.client_model.load_state_dict(model_params)

    def train(self):
        sigma = 0.01
        decay = 0.95
        for local_epoch in range(self.num_epochs):
            # local batch training
            train(model=self.client_model,
                device=self.device,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                sigma=sigma)
            sigma *= decay
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay


        low_rank_model = self.get_low_rank_params()
        # low_rank_model = self.client_model

        ##UDP
        client_params = self.client_model.state_dict()
        client_params = copy.deepcopy(self.client_model.state_dict())
        new_client_params = add_dp_noise(self.global_params, client_params, 1.0, 5, 1e-5)




        # return self.client_id, len(self.train_loader.dataset), self.client_model.state_dict()
        return self.client_id, len(self.train_loader.dataset), self.client_model.state_dict(), low_rank_model
        # return self.client_id, len(self.train_loader.dataset), new_client_params, low_rank_model

    def evaluate(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'valid']
        # return test accuracy of this client
        self.client_model.eval()
        if set_to_use == 'train':
            loader = self.train_loader
        elif set_to_use == 'test':
            loader = self.test_loader
        else:  # valid
            loader = self.valid_loader
        correct = evaluate(model=self.client_model, device=self.device, eval_loader=loader)
        return correct, len(loader.dataset)

    def get_low_rank_params(self):

        rank_ratio = 0.95
        rank = 100
        raw_model = copy.deepcopy(self.client_model.state_dict())
        # low_rank_model = copy.deepcopy(self.client_model.state_dict())
        first_weight_processed = False
        low_rank_model = OrderedDict()
        self.init_ratio = max(0.6, self.init_ratio - 0.01)
        if_Conv = False
        for key, value in self.client_model.named_parameters():
            if "weight" in key and not first_weight_processed:
                if "conv" in key:
                    if_Conv = True
                low_rank_model[key] = low_rank_decomposition_svd(value, rank, self.init_ratio, if_Conv)
                first_weight_processed = True
                if_Conv = False
            else:
                low_rank_model[key] = value.detach()
        return low_rank_model



@dmgan_class
class GanClient(Client):
    """

    Adversarial client adopting GAN attack

    """

    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 generator=None,
                 tracked_class=3,
                 local_epochs=1,
                 rec_epochs=1,
                 lr=0.1,
                 lr_decay=0.95,
                 momentum=0.5,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 dummy=None,
                 device=None):
        super(GanClient, self).__init__(
            client_id=client_id,
            client_group=client_group,
            client_model=client_model,
            local_epochs=local_epochs,
            lr=lr,
            lr_decay=lr_decay,
            momentum=momentum,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        self.tracked_class = tracked_class

        self.generator = generator.to(self.device)
        self.d_optimizer = optim.SGD(self.client_model.parameters(), lr=1e-3, weight_decay=1e-7)
        self.g_optimizer = optim.SGD(self.generator.parameters(), lr=1e-4, weight_decay=1e-7)

        self.rec_epochs = rec_epochs
        self.dummy = dummy



class terngradClient(Client):
    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 graient=None,
                 local_epochs=1,
                 lr=0.1,
                 lr_decay=0.95,
                 momentum=0.5,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 device=None):
        super(terngradClient, self).__init__(
            client_id=client_id,
            client_group=client_group,
            client_model=client_model,
            local_epochs=local_epochs,
            lr=lr,
            lr_decay=lr_decay,
            momentum=momentum,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        self.gradient = graient


    def synchronize(self, cur_round, model_params):
        self.client_model.load_state_dict(model_params)

    def calculate_gradient(self, raw_params):
        diffs = [((raw_params[k]-v) / self.lr).detach() for k, v in self.client_model.named_parameters()]
        return diffs

    def train(self):
        raw_params = copy.deepcopy(self.client_model.state_dict())
        for local_epoch in range(self.num_epochs):
            # local batch training
            train(model=self.client_model,
                  device=self.device,
                  train_loader=self.train_loader,
                  optimizer=self.optimizer,
                  criterion=self.criterion)
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay
        self.gradient = self.calculate_gradient(raw_params)
        self.ternarize_gradient()
        return self.client_id, len(self.train_loader.dataset), self.gradient, self.client_model.state_dict()

    def evaluate(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'valid']
            # return test accuracy of this client
        self.client_model.eval()
        if set_to_use == 'train':
            loader = self.train_loader
        elif set_to_use == 'test':
            loader = self.test_loader
        else:  # valid
            loader = self.valid_loader
        correct = evaluate(model=self.client_model, device=self.device, eval_loader=loader)
        return correct, len(loader.dataset)

    def ternarize_gradient(self):
        for key, value in self.client_model.named_parameters():
            self.client_model[key] = np.sign(value) * (np.abs(value) >= (np.max(np.abs(value)) / 2))