import torch
from torch.autograd import Variable

import copy
import logging

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

class LogisticRegressionOptimizer(Optimizer):
    def __init__(self, lr):
        super(LogisticRegressionOptimizer, self).__init__()
        self.lr = lr
        self.step = 0

    def _build_model(self, dim):
        self._model = Variable(torch.FloatTensor(dim).zero_())

    def minimize(self, dataset):
        self._build_model(dataset.num_features)
        y = Variable(torch.FloatTensor(dataset.labels))
        h = Variable(torch.FloatTensor(dataset.num_tuples).zero_())
        f_cur = Variable(torch.FloatTensor(dataset.num_tuples).zero_())

        r_prev = 0
        F = 0
        # iterative process
        while self.step <= 100000:
            for i in range(dataset.num_features):
                # compute partial grad first:
                col = Variable(torch.FloatTensor(dataset.fetch_col(i)))

                mul_arr = self._gradient_kl(col, y)

                f_partial = torch.sum(mul_arr)
                _prev_model = self._model[i].clone()
                # model partial update  
                self._model[i] = self._model[i] - self.lr * f_partial
                model_diff = self._model[i] - _prev_model
                # back_kl
                h = h + model_diff * col

            f_cur = self._loss_kl(y, h)
            r_curr = F
            F = torch.sum(f_cur)
            logger.info("Current Step: {}, Loss Value: {}".format(self.step, F.data.numpy()[0]))
            self.step += 1

    def _gradient_kl(self, y, col):
        return -y / (1 + torch.exp(col * y)) * col

    def _loss_kl(self, y, h):
        return torch.log(1 + torch.exp(-y * h))

