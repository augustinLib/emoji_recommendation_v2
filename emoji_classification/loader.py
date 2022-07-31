import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()

    
    def __len__(self):
        return self.data.size(0)


    def __getitem__(self, index):
        
        x = self.data[index]
        y = self.labels[index]

        return x, y


def load_senitimental_data():
    dataset = pd.read_csv("../data/sentimental.csv")

    x = dataset["sentences"]
    y = dataset["sentiment"]

    return x, y


def get_loaders(config):
    x, y = load_senitimental_data()

    train_cnt = int(x.size(0) * config.train_ratio) 
    valid_cnt = int(x.size(0) * config.vaild_ratio) 
    test_cnt = x.size(0) - train_cnt - valid_cnt

    indices = torch.randperm(x.size(0))

    train_x, valid_x, test_x = torch.index_select(
        x, 
        dim=0, 
        index=indices
    ).split([train_cnt, valid_cnt, test_cnt], dim = 0)


    train_y, valid_y, test_y = torch.index_select(
        y, 
        dim=0, 
        index=indices
    ).split([train_cnt, valid_cnt, test_cnt], dim = 0)

    train_loader = DataLoader(
        dataset= MyDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset= MyDataset(valid_x, valid_y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset= MyDataset(test_x, test_y),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader

