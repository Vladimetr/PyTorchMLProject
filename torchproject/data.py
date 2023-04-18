"""
структура данных в PyTorch
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


class MyDataset(Dataset):
    def __init__(self, data_path, params:dict):
        self.data_path = data_path
        # ------ параметры данных -------
        # пути, форматы, SampleRate и т п
        # параметры препроцессинга
        self.params = params
        self.data = pd.read_csv(data_path, sep=' ')
        self.n_classes = params["n_classes"]

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
        path, label = self.data.iloc[i]

        # preprocess

        T, F = 10, 256
        x = 0.6 * torch.ones(T, F)

        return x, int(label)
    
    def get_model_input(self, batch_size=1) -> dict:
        """
        Get dummy model input
        Returns:
            dict: kwargs for Model
        NOTE: don't forget about device matching btwn tensors and Model
        """
        batch_item = self.__getitem__(0)
        xs, ys = self.collate([batch_item] * batch_size)
        return {'x': xs}
    
    def collate(self, batch):
        """
        get torch data Tensors from batch list
        :param batch: list of (x, target)
                    as from __getitem__
        :return: x - torch Tensor
                y - torch Tensor
        """
        n = len(batch)

        MAX_T, FDIM = 100, 256
        xs = torch.zeros(n, MAX_T, FDIM)
        ys = []
        for i, (x, y) in enumerate(batch):
            t, f = x.shape
            xs[i, :t, :] = x
            ys.append(y)

        ys = torch.tensor(ys)
        ys = F.one_hot(ys, num_classes=self.n_classes)
        return xs, ys



if __name__ == '__main__':
    data_path = 'data/manifest.csv'
    params = {
        'sample_rate': 16000,
        'n_classes': 2,

    }
    dataset = MyDataset(data_path, params)
    sampler = BucketingSampler(dataset, batch_size=10, shuffle=True)
    test = CudaDataLoader(dataset=dataset, 
                          collate_fn=dataset.collate, 
                          batch_sampler=sampler,
                          pin_memory=True,
                          num_workers=1
    )
    