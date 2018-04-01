import numpy as np
import logging

_NUM_TUPLES = int(1e6)
_NUM_FEATS = int(9)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_data(file):
    labels = np.zeros(_NUM_TUPLES).astype(np.float32)
    # only features contains in raw
    raw = np.zeros((_NUM_TUPLES, _NUM_FEATS))
    with open(file, 'rb') as f:
        for i,line in enumerate(f.readlines()):
            processed = line.rstrip('\n').split(' ')
            labels[i] = processed[1]
            try:
                assert len(raw[i,:]) == len(processed[2:])
                raw[i,:] = processed[2:]
            except AssertionError as err:
                logger.info("Wrong Feature Number claimed !")
        return raw, labels


class Dataset(object):
    def __init__(self, data):
        # m, n denote number of tuples and features respectively
        self._m = data[0].shape[0]
        self._n = data[0].shape[1]
        self._training_data = data[0]
        self._training_labels = data[1]

    def fetch_col(self, col_index):
        return self._training_data[:, col_index]

    @property
    def num_tuples(self):
        return self._m

    @property
    def num_features(self):
        return self._n

    @property
    def labels(self):
        return self._training_labels


if __name__ == "__main__":
    data = load_data("T_float")
    dataset = Dataset(data)
    print('Training set shape: {}'.format((dataset.num_tuples,dataset.num_features)))
    print('Training label shape: {}'.format(dataset.num_tuples))
