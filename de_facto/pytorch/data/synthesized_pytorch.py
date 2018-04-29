from torch.utils.data import Dataset

class SynthesizedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset):
        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.root_dir = root_dir
        #self.transform = transform
        self._data = dataset

    def __len__(self):
        return self._data.num_tuples

    def __getitem__(self, index):
        data, label = self._data.data_table[index, :], self._data.labels[index]
        return data, label