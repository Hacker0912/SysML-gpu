import torch
from torch.autograd import Variable
from torch import nn

import time
import copy
import logging

from .utils import warn
from .optim import Optimizer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self, input_size, num_classes=1):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        h = self.fc(x)
        return h


class SCDOpimizer(Optimizer):
    def __init__(self, **args):
        super(SCDOpimizer, self).__init__()
        self.lr = args['lr']
        self.step = 0

        self._enable_gpu = args['enable_gpu']
        self._max_steps = args['max_steps']
        self._load_memory = args['load_memory']

    def _build_model(self, dim):
        if self._enable_gpu:
            self._model = Variable(torch.FloatTensor(dim).zero_()).cuda()
        else:
            self._model = Variable(torch.FloatTensor(dim).zero_())

    def minimize(self, dataset):
        self._build_model(dataset.num_features)

        load_memory_start = time.time()
        # load the entire dataset into memory
        if self._load_memory:
            self._load_data_in_memory(dataset)

        # init variables in CPU/GPU memory
        if self._enable_gpu:
            y = Variable(torch.FloatTensor(dataset.labels)).cuda()
            h = Variable(torch.FloatTensor(dataset.num_tuples).zero_()).cuda()
            f_cur = Variable(torch.FloatTensor(dataset.num_tuples).zero_()).cuda()
        else:
            y = Variable(torch.FloatTensor(dataset.labels))
            h = Variable(torch.FloatTensor(dataset.num_tuples).zero_())
            f_cur = Variable(torch.FloatTensor(dataset.num_tuples).zero_())   
        load_memory_duration = time.time() - load_memory_start

        print("Init Memory Movement time: {}".format(load_memory_duration))

        r_prev = 0
        F = 0

        # iterative process
        while self.step < self._max_steps:
            iter_start = time.time()
            memory_move_duration = 0
            reduce_duration = 0
            comp_duration = 0
            for i in range(dataset.num_features):
                # compute partial grad first:
                tmp_data_fetch_start = time.time()
                col = self._fetch_col(dataset, i)
                tmp_data_fetch_duration = time.time() - tmp_data_fetch_start
                memory_move_duration += tmp_data_fetch_duration

                # count GPU copy time
                tmp_comp_start = time.time()
                mul_arr = self._gradient_kl(y, col, h)

                tmp_reduce_start = time.time()
                f_partial = torch.sum(mul_arr)/dataset.num_tuples
                tmp_reduce_duration = time.time() - tmp_reduce_start
                reduce_duration += tmp_reduce_duration

                _prev_model = self._model[i].clone()

                # model partial update 
                self._model[i] = self._model[i] - self.lr * f_partial
                model_diff = self._model[i] - _prev_model
                
                h = h + model_diff * col
                tmp_comp_duration = time.time() - tmp_comp_start
                comp_duration += tmp_comp_duration

            tmp_comp_start2 = time.time()
            f_cur = self._loss_kl(y, h)/dataset.num_tuples
            r_curr = F

            tmp_reduce_start2 = time.time()
            F = torch.sum(f_cur)
            tmp_reduce_duration2 = time.time() - tmp_reduce_start2
            reduce_duration += tmp_reduce_duration2

            if self._enable_gpu:
                tmp_gpu_copy_start2 = time.time()
                _loss_val = F.cpu().data.numpy()[0]
                tmp_gpu_copy_duration2 = time.time() - tmp_gpu_copy_start2
                memory_move_duration += tmp_gpu_copy_duration2
            else:
                _loss_val = F.data.numpy()[0]
            
            tmp_comp_duration2 = time.time() - tmp_comp_start2
            comp_duration += tmp_comp_duration2

            logger_format = "Step: {}, Loss: {:.4f}, Iteration Cost: {:.4f}, Memory Move: {:.4f}, Comp Cost: {:.4f}, Reduce Cost: {:.4f}"
            print(logger_format.format(self.step, _loss_val, time.time()-iter_start, memory_move_duration, comp_duration, reduce_duration))
            self.step += 1

    def _gradient_kl(self, y, col, h):
        # 0-1 loss
        sign_loss = torch.sign(y * h - 1)
        negative_indices = (sign_loss < 0).nonzero().view(-1)
        if len(negative_indices) != 0:
            sign_loss.index_fill_(-1, negative_indices, 0)
        return sign_loss * col

    def _loss_kl(self, y, h):
        if self._enable_gpu:
            return torch.max(Variable(torch.zeros(y.size())).cuda(), 1 - y * h)
        else:
            return torch.max(Variable(torch.zeros(y.size())), 1 - y * h)

    def _load_data_in_memory(self, dataset):
        if self._enable_gpu:
            self._in_memory_dataset = Variable(torch.FloatTensor(dataset.data_table)).cuda()
        else:
            self._in_memory_dataset = Variable(torch.FloatTensor(dataset.data_table))

    def _fetch_col(self, dataset, index):
        if self._load_memory:
            return self._in_memory_dataset[:, index]
        else:
            if self._enable_gpu:
                return Variable(torch.FloatTensor(dataset.fetch_col(index))).cuda()
            else:
                return Variable(torch.FloatTensor(dataset.fetch_col(index)))

    def _gpu_memory_sanity_check(self):
        if self._load_memory and not self._enable_gpu:
            warn("######Shouldn't call loading dataset in GPU memory when not enabling GPU,\nAutomatically disable memory loading for safety########")
            self._load_memory = False


class SGDOpimizer(Optimizer):
    def __init__(self, **args):
        super(SGDOpimizer, self).__init__()
        self.lr = args['lr']
        self.step = 0

        self._enable_gpu = args['enable_gpu']
        self._max_steps = args['max_steps']

        self._C = 0.01

    def _build_model(self, dim):
        self._model = LinearSVM(input_size=dim)

        if self._enable_gpu:
            self._model.cuda() 

    def minimize(self, dataset, train_loader):
        self._build_model(dataset.num_features)
        for i, (data, labels) in enumerate(train_loader):
            iter_start = time.time()

            gpu_copy_start = time.time()
            if self._enable_gpu:
                data = Variable(torch.FloatTensor(data)).cuda()
                labels = Variable(torch.FloatTensor(labels)).cuda()
            else:
                data = Variable(torch.FloatTensor(data))
                labels = Variable(torch.FloatTensor(labels))
            gpu_copy_duration = time.time() - gpu_copy_start
            
            # Forward + Backward + Optimize
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr)
            self._optimizer.zero_grad()

            fwd_start = time.time()
            outputs = self._model(data)
            fwd_dur = time.time() - fwd_start

            loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))  # hinge loss
            loss += self._C * torch.mean(self._model.fc.weight**2)  # l2 penalty

            bwd_start = time.time()
            loss.backward()
            bwd_dur = time.time() - bwd_start

            update_start = time.time()
            self._optimizer.step()
            update_dur = time.time() - update_start

            logger_format = "Step: {}, Loss: {:.4f}, Iteration Cost: {:.4f}, GPU Copy: {:.4f}, Forward: {: .4f}, Backward: {: .4f}, Update: {: .4f}"
            logger.info(logger_format.format(i, loss.data[0], time.time()-iter_start, gpu_copy_duration, 
                                           fwd_dur, bwd_dur, update_dur))
