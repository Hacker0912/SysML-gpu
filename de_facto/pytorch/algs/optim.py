class Optimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def minimize(self, *args, **kwargs):
        raise NotImplementedError()

    def _stop(self, *args, **kwargs):
        raise NotImplementedError()