import torch
from torch.utils.data import Dataset
import torchvision

class MNIST_dataset(Dataset):
    def __init__(self, root_dir, train: bool = True):
        dataset = torchvision.datasets.MNIST(root=root_dir, train=train, download=True)
        ## From training set
        self.images = torch.as_tensor(dataset.data, dtype=torch.float) / 127.5 - 1.0
        self.images = torch.einsum("bwh -> bhw", self.images)
        self.labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        item = {}
        item['image'] = torch.unsqueeze(self.images[idx], 0)
        item['y'] = self.labels[idx]
        return item