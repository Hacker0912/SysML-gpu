import torch
from torch.autograd import Variable

class Optimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def minimize(self, *args, **kwargs):
        raise NotImplementedError()

class LogisticRegressionOptimizer(Optimizer):
    def __init__(self, lr):
        super(LogisticRegressionOptimizer, self).__init__()
        self.lr = lr

    def minimize(self, dataset):
        y = Variable(torch.FloatTensor(dataset.labels))
        for i in range(dataset.num_features):
            # compute partial grad first:
            col = Variable(torch.FloatTensor(dataset.fetch_col(i)))
            print(dataset.fetch_col(i))
            print('=================================================')
            exit()
            tmp = col * y
            print(tmp)
            exit()