from abc import ABCMeta, abstractmethod
import os.path as osp
from typing import Iterable, List
import pandas as pd
from tqdm import tqdm
from .. import utils
from ..utils.manager import ClearMLDataset
from .dataset import NoiseClassificationDataset, BucketingSampler, CudaDataLoader


COLUMNS = ["audio", "start", "end", "label"]
SR = 16000  # sample rate
CSV_SEP = ","

"""
audio - /abs/path/to/audio.wav
start - start sample (not sec)
end - end sample (not sec)
label - name of label (str)
"""

    
class DataProcess(metaclass=ABCMeta):
    name = "some-process"
    def __init__(self, **kwargs):
        """
        **kwargs - additional params for process
        """
        self.data : pd.DataFrame = None
        self.meta : dict = None
        self.in_paths : List[str] = []  # [input manifest.csv]
        # it can be multible for specific tasks like join
        self.out_path : str = None  # output manifest.csv
        self.kwargs = kwargs  # additional params for processing
        # don't forget super().__init__(**kwargs) for child classes

    def _get_progress_bar(self, iter: Iterable) -> Iterable:
        bar_fmt = '| {n_fmt}/{total_fmt} {postfix}'
        iter = tqdm(iter, 
                    desc="process", 
                    total=len(iter),
                    bar_format='{l_bar}{bar:29}' + bar_fmt)
        return iter

    def _load_data(self, manifest_csv:str):
        """ from manifest define
        self.data
        self.data_size
        """
        self.data = pd.read_csv(manifest_csv, sep=CSV_SEP)
        self.data_size = len(self.data)
    
    def _set_metadata(self):
        """ define self.meta from self.data
        """
        if self.data is None:
            raise ValueError("self.data is not defined. Load it first")
        meta = {
            "data_size": len(self.data),
            "n_classes": len(self.data['label'].unique()),
            "sample_dur": utils.get_sample_dur(self.data, SR)
        }
        self.meta = meta
    
    def _save_data(self, data_path: str, overwrite=False):
        """
        NOTE: if data_path exists and overwrite=False,
            next version (v2->v3) will be created
        """
        if osp.exists(data_path) and not overwrite:
            data_path = utils.get_next_version(data_path)
        self.data.to_csv(data_path, sep=CSV_SEP,
                         header=True, index=False)
        return data_path
    
    @abstractmethod
    def process(self, input, output):
        """
        input/output can vary depending on task
        """
        raise NotImplementedError("abstract")
    
    def add_manager_info(self, 
                         tags:List[str]=None, 
                         description:str=None,
                         version:str=None,
                         commit:bool=True):
        """ 
        Create new unique dataset in ClearML
        Args:
            tags (List[str]): custom tags
            description (str): custom description
                Data path by default
            version (str): custom version in format
                '1.2-postfix'
            commit (bool): whether to finalize (commit)
                version in ClearML. Set False, if you want
                to add something else further
        """
        if not self.out_path:
            print("Data is not saved. Can not log to ClearML")
            return
        print("Writing to ClearML...", flush=True)
        # check already exists
        exist_dataset = ClearMLDataset.get_by_data(self.out_path)
        if exist_dataset:
            raise ValueError(f"ClearML dataset with name {self.out_path} "\
                              "already exists")

        # define parent(s)
        prevs = []
        for inp in self.in_paths:
            prev = ClearMLDataset.get_by_data(inp)
            if prev:
                prevs.append(prev.id)

        # tags = user tags + task name 
        if not tags:  tags = []                
        if self.name:
            tags.append(self.name)
        if not self.meta:
            self._set_metadata()

        # properties = meta + kwargs
        properties = {**self.meta, **self.kwargs}

        description = description or self.out_path

        self.manager = ClearMLDataset.create(self.out_path, 
                                            previous=prevs,
                                            description=description,
                                            version=version,
                                            tags=tags,
                                            **properties)
        if commit:
            self.manager.commit()
   