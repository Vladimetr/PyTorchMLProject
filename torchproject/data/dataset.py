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
from ..preprocess import init_preprocessor, load_audio
from .. import utils

CSV_SEP = ' '


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


class AntispoofDataset(Dataset):
    def __init__(self, data_path:str, classes:List[str], 
                 sr:int=8000,
                 preprocess_cfg:dict=None,
                 ):
        """
        data_path (str): /path/to/manifest.csv
        |--audio--|--start--|--end--|--class--|
        """
        self.data_path = data_path
        print(f"Loading manifest '{data_path}'...")
        data = pd.read_csv(data_path, sep=CSV_SEP)
        self.classes = classes
        self.n_classes = len(classes)

        if preprocess_cfg:
            preprocess_cfg = dict(preprocess_cfg)  # copy
            self.preprocessor = init_preprocessor(preprocess_cfg)
        else:
            self.preprocessor = None

        self.data = data
        self.sr = sr

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
              (B, F, T): model input
              int: label
        """
        audio_path, start, end, class_name = self.data.iloc[i]

        # load audio
        sample, _ = load_audio(audio_path)
        # (1, S)

        # cut
        x = sample[:1, int(self.sr * start) : int(self.sr * end)]
        # (1, S')

        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)

        # label (str) -> (int)
        try:
            label = self.classes.index(class_name)
        except ValueError:
            raise ValueError(f"Invalid label name '{class_name}'")

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
            (B, F, T) or (B, 1, S): batch of inputs
            (B, ): batch of labels
        """
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)  # (1, S)
            ys.append(y)  # int
            
        xs = torch.stack(xs, dim=0)  # (B, 1, S)
        ys = torch.tensor(ys, dtype=torch.long)  # (B, )
        return xs, ys



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    preprocess_cfg = config["preprocess"]
    data_path = 'data/processed/train_manifest.v1.csv'

    dataset = AntispoofDataset(data_path, classes=config["classes"], 
                               preprocess_cfg=preprocess_cfg)
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
    