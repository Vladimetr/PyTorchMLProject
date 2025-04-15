"""
Data batching for train and test
"""
from typing import List
import os.path as osp
import numpy as np
import pandas as pd
import random
import sys
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from ..preprocess import init_preprocessor
from .. import utils

SR = 16000
CSV_SEP = ","


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
    If shuffle:
    1) batches are shuffled among each other
    2) samples are shuffled inside each batch
    [ [8, 1, 4], [5, 2, 9], [3, 6, 7], ... ]
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
        """ Number of batches
        """
        return len(self.bins[:self.limit])

    def shuffle(self, epoch):
        """ Shuffle batches (every epoch)
        """
        if self.do_shuffle:
            np.random.RandomState(epoch).shuffle(self.bins)


class NoiseClassificationDataset(Dataset):
    def __init__(self, data_path:str, classes:List[str], 
                 sr:int=SR, normalize:bool=True,
                 preprocess_cfg:dict=None,
                 cache_size:int=0
                 ):
        """
        data_path (str): /path/to/manifest.csv
            |--audio--|--start--|--end--|--class--|
        preprocess_cfg (dict, None): if None, source samples
            are stored in batch (B, 1, S)
        cache_size (int): number of audio samples stored in RAM
            for faster batch generating
        """
        self.data_path = data_path
        print(f"Loading manifest '{data_path}'...")
        data = pd.read_csv(data_path, sep=CSV_SEP)
        self.classes = classes
        self.n_classes = len(classes)
        self.normalize = normalize

        if preprocess_cfg:
            preprocess_cfg = dict(preprocess_cfg)  # copy
            self.preprocessor = init_preprocessor(preprocess_cfg)
        else:
            self.preprocessor = None

        self.data = data
        self.sr = sr
        self.cache_samples = dict()  # {"audio_path": sample (1, S)}
        self.cache_size = cache_size

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

        # try load audio from cache first
        try:
            sample = self.cache_samples[audio_path]
        except KeyError:
            # torchaudio load
            sample, _ = utils.load_audio(audio_path, self.normalize)
            # put in cache
            if len(self.cache_samples) > self.cache_size:
                self.cache_samples.clear()
            self.cache_samples[audio_path] = sample
        # (1, S)

        # cut
        x = sample[:1, start : end]
        # (1, S')
        x = torch.unsqueeze(x, 0)
        # (B=1, 1, S)

        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)
            # (B, F, T)

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
            xs.append(x)  # (1, 1, S) or (1, F, T)
            ys.append(y)  # int
            
        xs = torch.cat(xs, dim=0)  # (B, 1, S)
        ys = torch.tensor(ys, dtype=torch.long)  # (B, )
        return xs, ys
    

def validate(manifest:str, dur:float, classes:List[str],
             sr:int=SR):
    """ Validation of manifest
    - audios exist
    - duration is equal to given
    - class is valid
    It's good to run it before training
    Args:
        manifest (str): /path/to/manifest.csv
        dur (float): duration in sec of single classifed sample (row)
        classes (List[str]): list of valid classes
        sr (int): sample rate
    """
    # read data
    data = pd.read_csv(manifest, sep=CSV_SEP)
    if isinstance(classes, str):
        # read classes from file
        classes = utils.read_classes(classes)

    prev_wavpath = None
    dur = int(sr * dur)
    for i in tqdm(range(len(data))):
        wavpath, start, end, label = data.iloc[i]

        # load audio
        if wavpath != prev_wavpath:
            sample, sr = utils.load_audio(wavpath)
            assert sr == SR, wavpath
            slen = sample.shape[1]
            prev_wavpath = wavpath

        assert end <= slen, f"row {i}"
        assert end - start == dur, f"row {i}"

        assert label in classes, f"label: '{label}', row {i}"

    print("Data is OK")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset validation')
    parser.add_argument('--config', '-cfg', type=str, 
                        default="config.yaml",
                        help='path/to/config.yaml')
    parser.add_argument('--input', '-i', type=str, 
                        required=True,
                        help='path/to/input/data.csv')
    args = parser.parse_args()

    validate(args.input, dur=5.0, classes="classes/fsd50.txt")
    exit()

    config = utils.config_from_yaml(args.config)
    classes = utils.read_classes(config["classes"])
    preprocess_cfg = config["preprocess"]

    train_set = NoiseClassificationDataset(args.input, classes=classes,
                                 sr=SR, cache_size=1000,
                                 preprocess_cfg=preprocess_cfg)
    sampler = BucketingSampler(train_set, 4,
                               shuffle=False)
    train_set = CudaDataLoader(0, train_set, 
                               collate_fn=train_set.collate, 
                               pin_memory=True, num_workers=8,
                               batch_sampler=sampler)
    
    for i, batch in enumerate(train_set):
        # i, x, y  
        print(i, batch[0].shape, batch[1].shape)

    