import torch
from torch.autograd import Variable

import time
import copy
import logging

from .utils import warn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Optimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def minimize(self, *args, **kwargs):
        raise NotImplementedError()

    def _stop(self, *args, **kwargs):
        raise NotImplementedError()

class SCDOptimizer(Optimizer):
    def __init__(self, **args):
        super(SCDOptimizer, self).__init__()
        self.lr = args['lr']
        self.step = 0

        self._enable_gpu = args['enable_gpu']
        self._max_steps = args['max_steps']
        self._load_memory = args['load_memory']
        self._gpu_memory_sanity_check()

    def _build_model(self, dim):
        if self._enable_gpu:
            self._model = Variable(torch.FloatTensor(dim).zero_()).cuda()
        else:
            self._model = Variable(torch.FloatTensor(dim).zero_())

    def minimize(self, dataset):
        self._build_model(dataset.num_features)

        if self._enable_gpu:
            gpu_copy_base_start = time.time()
            if self._load_memory:
                self._load_data_in_memory(dataset)
            y = Variable(torch.FloatTensor(dataset.labels)).cuda()
            h = Variable(torch.FloatTensor(dataset.num_tuples).zero_()).cuda()
            f_cur = Variable(torch.FloatTensor(dataset.num_tuples).zero_()).cuda()
            gpu_copy_base_duration = time.time()-gpu_copy_base_start
        else:
            y = Variable(torch.FloatTensor(dataset.labels))
            h = Variable(torch.FloatTensor(dataset.num_tuples).zero_())
            f_cur = Variable(torch.FloatTensor(dataset.num_tuples).zero_())            

        r_prev = 0
        F = 0
        # iterative process
        while self.step <= self._max_steps:
            iter_start = time.time()
            for i in range(dataset.num_features):
                # compute partial grad first:
                if self._enable_gpu:
                    gpu_iter_copy_start = time.time()
                    col = self._fetch_col(i)
                    gpu_copy_duration = time.time()-gpu_iter_copy_start+gpu_copy_base_duration
                else:
                    col = Variable(torch.FloatTensor(dataset.fetch_col(i)))
                    gpu_copy_duration = 0

                grad_comp_start = time.time()
                mul_arr = self._gradient_kl(y, col, h)
                f_partial = torch.sum(mul_arr)
                _prev_model = self._model[i].clone()
                grad_comp_duration = time.time() - grad_comp_start

                # model partial update
                model_update_start = time.time() 
                self._model[i] = self._model[i] - self.lr * f_partial
                model_diff = self._model[i] - _prev_model
                model_update_duration = time.time() - model_update_start
                
                # back_kl
                backkl_start = time.time()
                h = h + model_diff * col
                backkl_duration = time.time() - backkl_start

            loss_comp_start = time.time()
            f_cur = self._loss_kl(y, h)
            r_curr = F
            F = torch.sum(f_cur)
            loss_comp_duration = time.time() - loss_comp_start
            if self._enable_gpu:
                _loss_val = F.cpu().data.numpy()[0]
            else:
                _loss_val = F.data.numpy()[0]
            logger_format = "Step: {}, Loss: {:.4f}, Iteration Cost: {:.4f}, GPU Copy: {:.4f}, Grad Comp: {:.4f}, Model Update: {:.4f}, Backkl: {:.4f}, Loss Comp: {:.4f}"
            logger.info(logger_format.format(self.step, _loss_val, time.time()-iter_start, gpu_copy_duration, 
                                            grad_comp_duration, model_update_duration, backkl_duration, loss_comp_duration))
            self.step += 1

    def _gradient_kl(self, y, col, h):
        return -y / (1 + torch.exp(h * y)) * col

    def _loss_kl(self, y, h):
        return torch.log(1 + torch.exp(-y * h))

    def _load_data_in_memory(self, dataset):
        self._in_memory_dataset = Variable(torch.FloatTensor(dataset.data_table)).cuda()

    def _fetch_col(self, index):
        if self._load_memory:
            return self._in_memory_dataset[:, index]
        else:
            return Variable(torch.FloatTensor(dataset.fetch_col(i))).cuda()

    def _gpu_memory_sanity_check(self):
        if self._load_memory and not self._enable_gpu:
            warn("######Shouldn't call loading dataset in GPU memory when not enabling GPU,\nAutomatically disable memory loading for safety########")
            self._load_memory = False


class SGDOptimizer(Optimizer):
    def __init__(self, lr):
        super(SGDOptimizer, self).__init__()
        self.lr = lr
        self.step = 0

    def _build_model(self, dim):
        self._model = Variable(torch.FloatTensor(dim).zero_())

    def minimize(self, dataset):
        self._build_model(dataset.num_features)
        # shuffle dataset
        dataset.shuffle()