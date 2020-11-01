import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, size=1000, noise_rate=0.0, seed=0, train=True):
        torch.manual_seed(seed)
        self.size = 2 * size
        label0 = torch.normal(mean=torch.zeros(size, 2) + torch.Tensor([0,2]), std=torch.ones(size, 2))
        label1 = torch.normal(mean=torch.zeros(size, 2) + torch.Tensor([2,0]), std=torch.ones(size, 2))
        self.data = torch.cat((label0, label1))
        self.labels = torch.cat((torch.zeros(size), torch.ones(size))).type(torch.LongTensor)
        if (not train):
            label0 = torch.normal(mean=torch.zeros(size, 2) + torch.Tensor([0,2]), std=torch.ones(size, 2))
            label1 = torch.normal(mean=torch.zeros(size, 2) + torch.Tensor([2,0]), std=torch.ones(size, 2))
            self.data = torch.cat((label0, label1))
        else:
            flip_indices = torch.randperm(2 * size)[:int(2 * noise_rate * size)]
            self.labels[flip_indices] = 1 - self.labels[flip_indices]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], idx)


if __name__ == '__main__':
    dataset = SyntheticDataset()
    print(len(dataset))
    print(dataset[100])
    print(dataset[995:1005])