"""
Data batching for train and test
"""
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import random
import sys
from collections import OrderedDict
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from ..preprocess import init_preprocessor
from .. import utils

CSV_SEP = ','


class CudaDataLoader(DataLoader):
    def __init__(self, gpu_id:int=0, *args, **kwargs):
        self.gpu_id = gpu_id
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Relocate data from CPU to GPU
        """
        for cpu in super().__iter__():
            gpu = []
            for values in cpu:
                if isinstance(values, Tensor):
                    gpu.append(values.contiguous().cuda(
                        self.gpu_id, non_blocking=True))
                else:
                    gpu.append(values)
            yield gpu

    def shuffle(self, epoch):
        """ Shuffle batches (every epoch)
        """
        self.batch_sampler.shuffle(epoch)

    def __len__(self) -> int:
        """ Number of batches
        """
        return len(self.batch_sampler)


class BucketingSampler(Sampler):
    """
    Organize batch indices
    For batch_sz = 3
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9], ... ]
    Then 
    1) batches are shuffled among each other
    2) samples are shuffled inside each batch
    """
    def __init__(self, dataset, batch_size, 
                 limit=sys.maxsize, shuffle=True):
        super().__init__(dataset)
        index = list(range(len(dataset)))  # [0, 1, 2, 3, ... n]
        if shuffle:
            random.shuffle(index)

        self.bins = [index[i:i + batch_size] \
                     for i in range(0, len(index), batch_size)]
        # [ [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], ... ]

        self.limit = limit
        self.do_shuffle = shuffle

    def __iter__(self):
        # выдать индексы батчей
        for batch in self.bins[:self.limit]:
            if self.do_shuffle:
                random.shuffle(batch)  # shuffle samples inside batch
            yield batch

    def __len__(self) -> int:
        """ Number of batched
        """
        return len(self.bins[:self.limit])

    def shuffle(self, epoch):
        """ Shuffle batches (every epoch)
        """
        if self.do_shuffle:
            np.random.RandomState(epoch).shuffle(self.bins)


class BlockChainDataset(Dataset):
    def __init__(self, data_path:str, classes:List[str]):
        """
        config (dict): see config.yaml for example
        """
        self.data_path = data_path
        print(f"Loading manifest '{data_path}'...")
        df = pd.read_csv(data_path, sep=CSV_SEP)
        self.data = df.to_dict(into=OrderedDict, orient='index')
        self.features_names = list(df.columns)
        self.features_names.remove("label")
        self.classes = classes
        self.n_classes = len(classes)
        self.df = df

    def get_features_names(self) -> List[str]:
        return self.features_names

    def get_data(self, device:str="cpu", shuffle=False
                 ) -> Tuple[Tensor, Tensor]:
        """
        NOTE: out-of-memory may appear
        """
        df = self.df
        if shuffle:
            df = df.sample(frac=1)

        print("Data extraction ...")
        xs = []  # [(F, )]
        for feature_name in self.features_names:
            feature = df[feature_name].tolist()
            xs.append(torch.tensor(feature))  
        xs = torch.stack(xs, dim=1).to(device)  # (M, F)
        xs = xs

        labels = df["label"].tolist()
        ys = []
        for label in labels:
            try:
                y = self.classes.index(label)
            except ValueError:
                raise ValueError(f"Invalid label name '{label}'")
            ys.append(y)
        ys = torch.tensor(ys).to(device)
        return xs, ys

    def __len__(self) -> int:
        """
        Data size
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        load i-th data sample  образец
        apply given preprocess
        Returns:
            tuple
              (F, ): feature vector
              int: label
        """
        features : dict = self.data[i]
        label : str = features.pop("label")
        assert list(features.keys()) == self.features_names
        
        # features must be preprocessed beforehand
        x = torch.tensor(list(features.values()))

        # label (str) -> (int)
        try:
            label = self.classes.index(label)
        except ValueError:
            raise ValueError(f"Invalid label name '{label}'")

        return x, label
    
    def get_model_input(self, batch_size=1) -> dict:
        """
        Get dummy model input
        Returns:
            dict: kwargs for Model
        NOTE: don't forget about device matching btwn tensors and Model
        """
        batch_item = self.__getitem__(0)
        xs, _ = self.collate([batch_item] * batch_size)
        return {'x': xs}
    
    def collate(self, batch:List[tuple]):
        """
        Get data tensors from batch list
        Args:
            batch (list[Tensor, int])
        Returns:
            (B, F): batch of inputs
            (B, ): batch of labels
        """
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)
            ys.append(y)
            
        xs = torch.stack(xs, dim=0)  # (B, F)
        ys = torch.tensor(ys, dtype=torch.long)  # (B)
        return xs, ys



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    preprocess_params = config["preprocess"]
    data_path = 'data/processed/test.v1.csv'

    dataset = BlockChainDataset(data_path, classes=config["classes"], 
                                preprocess_params=preprocess_params)
    sampler = BucketingSampler(dataset, batch_size=2, shuffle=True)
    dataset = CudaDataLoader(dataset=dataset, 
                          collate_fn=dataset.collate, 
                          batch_sampler=sampler,
                          pin_memory=True,
                          num_workers=1
    )
    for i, batch in enumerate(dataset):
        x, label = batch
        print(x.shape, label.shape)
    