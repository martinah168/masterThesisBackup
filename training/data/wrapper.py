from torch.utils.data import Dataset


class SubsetDataset(Dataset):

    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


class Repeat(Dataset):

    def __init__(self, dataset, new_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index):
        index = index % self.original_len
        return self.dataset[index]
