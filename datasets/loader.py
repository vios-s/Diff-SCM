import torch
from torch.utils.data import Dataset
import torchvision


def get_data_loader(path, batch_size, split_set: str = 'train', which_label: str = "class"):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    print(path)
    dataset = MNIST(root_dir=path, train=split_set != "test")

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]

    loader = torch.utils.data.DataLoader(dataset, **default_kwargs)
    while True:
        yield from loader


class MNIST(Dataset):
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
        item['class'] = self.labels[idx]
        return item