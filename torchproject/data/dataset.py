"""
Data batching for train and test
"""
import numpy as np
import pandas as pd
import random
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from ..preprocess import AudioPreprocess
from .. import utils


class CudaDataLoader(DataLoader):
    # принимает экземпляр класса Dataset и Sampler

    def __init__(self, gpu_id:int=0, *args, **kwargs):
        self.gpu_id = gpu_id
        super().__init__(*args, **kwargs)       # вызывает метод __init__
                                                # родительского класса

    def __iter__(self):
        # перенести батч с ЦПУ на ГПУ
        for cpu in super().__iter__():
            # cpu - list of tensors on CPU

            gpu = []  # list of tensors on GPU
            for values in cpu:
                # перед отправкой на ГПУ, надо преобразовать в contiguous()
                # if pin_memory==True -> non_blocking=True
                if isinstance(values, torch.Tensor):
                    gpu.append(values.contiguous().cuda(
                        self.gpu_id, non_blocking=True))
                else:
                    gpu.append(values)

            yield gpu

    def shuffle(self, epoch):
        # перемешать батчи между собой (каждую эпоху)
        self.batch_sampler.shuffle(epoch)

    def __len__(self):
        # кол-во батчей
        return len(self.batch_sampler)


class BucketingSampler(Sampler):
    """
    организует индексы
    при batch_sz = 3
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9], ... ]
    затем перемешиваются
    1) батчи между собой
    2) образцы внутри батча
    * также можно поставить ограничение на кол-во
    """
    def __init__(self, dataset, batch_size, limit=sys.maxsize, shuffle=True):
        """
        :papam data: экземпляр класса Dataset
        """
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
                random.shuffle(batch)       # перемешать образцы внутри батча
            yield batch

    def __len__(self):
        # кол-во батчей
        return len(self.bins[:self.limit])

    def shuffle(self, epoch):
        # перемешать батчи между собой (каждую эпоху)
        if self.do_shuffle:
            np.random.RandomState(epoch).shuffle(self.bins)


class AudioDataset(Dataset):
    def __init__(self, data_path:str, n_classes:int, preprocess_params:dict):
        """
        config (dict): see config.yaml for example
        """
        self.data_path = data_path
        preprocess_params = dict(preprocess_params)
        self.frame = preprocess_params.pop("frame")
        self.preprocess = AudioPreprocess(**preprocess_params)
        self.data = pd.read_csv(data_path, sep=' ')
        self.n_classes = n_classes

    def __len__(self):
        # размер всего датасета
        # полное кол-во всех образцов
        return len(self.data)

    def __getitem__(self, i):
        """
        подгружаем из диска
        i-ый образец
        из всего датасета,
        делаем препроцесс
        и
        отправляем в виде
        -- 1 образец --
        x_data, y_data
        """
        npy_path, shape, start, end, label = self.data.iloc[i]
        # load npy
        sample = np.memmap(npy_path, dtype='float32',
                           mode='r', shape=(shape, ))
        sample = torch.tensor(sample).view(1, -1)
        sample = sample[:, start:end]

        # features extraction
        features = self.preprocess.extract_features(sample)
        # (F, T)
        framed_features = self.preprocess.features_extractor.split(
                                features=features, chunk=self.frame
        )
        # (N, F, T)

        return framed_features, int(label)
    
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
    
    def collate(self, batch):
        """
        get torch data Tensors from batch list
        :param batch: list of (x, target)
                    as from __getitem__
        :return: x - torch Tensor
                y - torch Tensor
        """
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)
            ys.append(y)

        xs = torch.cat(xs, dim=0)
        ys = torch.tensor(ys)
        ys = F.one_hot(ys, num_classes=self.n_classes)
        return xs, ys



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    preprocess_params = config["preprocess"]
    data_path = 'data/processed/train_manifest.v1.csv'

    dataset = AudioDataset(data_path, n_classes=2, 
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
    