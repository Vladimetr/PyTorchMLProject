"""
Data pipeline: clean, preprocess, balancing, etc
with different data versions
"""
from os import path as osp
import re
from abc import ABCMeta
from typing import Iterable, Optional, List, Union
import argparse
import pandas as pd
from tqdm import tqdm
from ..utils import get_next_version, get_version_from_path
from ..utils.manager import ClearMLDataset


CSV_SEP = ','
VALID_LABELS = [
    "mixer",
    "game",
    "user"

]
data_type = Union[str, List[str], pd.DataFrame]


class DataProcessor(metaclass=ABCMeta):
    """
    Base data processor
    """
    def __init__(self, **kwargs):
        """
        name (str): name of processing. For ex. "remove-duplicates"
        **kwargs: additional params of processing
        """
        self.name = "some-process"

    def _get_progress_bar(self, iter: Iterable) -> Iterable:
        bar_fmt = '| {n_fmt}/{total_fmt} {postfix}'
        iter = tqdm(iter, 
                    desc="process", 
                    total=len(iter),
                    bar_format='{l_bar}{bar:29}' + bar_fmt)
        return iter

    def _load_data(self, data_path: str):
        self.data = pd.read_csv(data_path, sep=CSV_SEP)
        self.data_size = len(self.data)
        self.columns = self.data.columns

    def _load_datas(self, data_paths: List[str]):
        datas = []
        for data_path in data_paths:
            data = pd.read_csv(data_path, sep=CSV_SEP)
            datas.append(data)
        self.data = pd.concat(datas, ignore_index=True)
        self.data_size = len(self.data)
        self.columns = self.data.columns

    def _save_data(self, data_path: str, overwrite = False):
        """
        NOTE: if data_path exists and overwrite=False,
            next version (v2->v3) will be created
        """
        if osp.exists(data_path) and not overwrite:
            data_path = get_next_version(data_path)
        self.data.to_csv(data_path, sep=CSV_SEP,
                         header=True, index=False)
        return data_path
        
    def process_row(self, row: dict) -> dict:
        """
        Algorithm of processing manifest row
        Args:
            row (dict): dict {'<column>': <value>}
        Returns:
            dict: new {'<column>': <value>}
        NOTE: don't change columns, values only
        NOTE: if returned dict is empty, 
        this row will be deleted from manifest
        """
        raise NotImplementedError("abstract")

    def process_rows(self):
        """
        Aply method 'process_row' to each row in given manifest
        Args:
            manifest_in (str): path/to/input/manifest.csv
            manifest_out (str, None): path/to/output/manifest.csv
                If None, use next version *.v{x}.csv -> *.v{x+1}.csv
        """
        iters = self._get_progress_bar(range(self.data_size))
        delete_rows = []
        for i in iters:
            row = self.data.iloc[i].to_dict()
            upd_row = self.process_row(row)  # updated row
            if upd_row:
                assert list(row.keys()) == list(upd_row.keys()), \
                    f"Mismatch columns in row [{i}]"
                # update row
                self.data.loc[i] = [upd_row[k] for k in self.columns] 
            else:
                delete_rows.append(i)
        # delete rows
        self.data.drop(delete_rows, inplace=True)

    def process(self, data_in: data_type,
                save_path: Optional[str]=None):
        """
        Main method for running process
        Args:
            data_in: path or list[path] to manifest.csv
                or pd.Dataframe
            save_path (str, None): /path/to/manifest.csv
                If path exists, use next version (v2->v3)
                If None, no save
        Returns:
            tuple
              pd.DataFrame: processed data
              str: /path/to/saved/data.csv or None
        """
        print(f"Process task {self.name}...")
        if isinstance(data_in, str):
            print(f" -input: {data_in}")
            self._load_data(data_in)
        elif isinstance(data_in, list):
            print(f" -input: {data_in}")
            self._load_datas(data_in)
        elif isinstance(data_in, pd.DataFrame):
            print(f" -input: DataFrame")
            self.data = data_in.copy(deep=True)

        self.process_rows()

        if save_path:
            save_path = self._save_data(save_path, overwrite=False)
            print(f" -output: {save_path}")

        return self.data, save_path
        

class DropDuplicatesProcessor(DataProcessor):
    """
    Drop duplucated rows
    """
    def __init__(self):
        self.name = 'drop-duplicates'

    def process_rows(self):
        self.data.drop_duplicates(inplace=True)
        

class DropInvalidLabelsProcessor(DataProcessor):
    """
    Remove rows if value in column 'label' not in 
    VALID_LABELS
    """
    def __init__(self):
        self.name = 'drop-invalid-labels'

    def process_row(self, row: dict) -> dict:
        label = row["label"]
        if not label in VALID_LABELS:
            return {}
        return row

class RemoveColumnsProcessor(DataProcessor):
    """
    Remove specific set of columns
    """
    def __init__(self, columns:List[str]):
        self.name = 'drop-columns'
        self.drop_columns = columns

    def process_rows(self):
        self.data.drop(self.columns, axis=1, inplace=True)


class JoinManifestsProcessor(DataProcessor):
    def __init__(self):
        self.name = 'join-manifests'

    def process_rows(self):
        pass

    def process(self, data_in: data_type,
                save_path: Optional[str]=None):
        # DataProcessor.process do the same
        return super().process(data_in, save_path)


TASKS = {
    "join-manifests": JoinManifestsProcessor,
    "drop-duplicates": DropDuplicatesProcessor,
    "drop-invalid-labels": DropInvalidLabelsProcessor,
    
}


def pipeline(input:Union[str, List[str]], tasks:List[str], output:str,
             save_steps:bool=False) -> pd.DataFrame:
    """
    Run pipeline of data processing
    Args:
        input (str, List[str]): path/to/data.csv or [path/to/data.csv]
            If list, collate them first
        tasks (list[str]): order is important. See TASKS
        output (str): where to save final output
        save_steps (bool): whether to save result for middle steps
    Returns:
        pd.DataFrame: final data
    """
    n = len(tasks)
    if not n:
        raise ValueError("List of tasks is empty")
    
    if isinstance(input, list) and tasks[0] != "join-manifests":
        # collate first
        tasks = ["join-manifests"] + tasks

    out_data = None
    for i, task in enumerate(tasks):
        try:
            processor : DataProcessor = TASKS[task]
        except KeyError:
            raise ValueError(f"Invalid task '{task}'")
        
        if i + 1 == n:
            # final task
            save_path = output
        elif save_steps is None:
            # without save
            save_path = None
        elif task == "join-manifests":
            save_path = input if isinstance(input, str) else \
                        input[0]

        out_data, _ = processor.process(out_data or input, 
                                        save_path=save_path)
    
    return out_data
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing functions')
    parser.add_argument('--task', '-t', type=str,
                        required=True,
                        choices=TASKS,
                        help='task to apply')
    parser.add_argument('--input', '-i', type=str, 
                        nargs='+',
                        required=True,
                        help='path/to/input/data(s).csv')
    parser.add_argument('--output', '-o', type=str, 
                        help='path/to/output/data.csv')
    parser.add_argument('--clearml', action='store_true', 
                        default=False, 
                        help='whether to use ClearML')
    parser.add_argument('--description', '-d', type=str, default=None, 
                        help='Description for ClearML')
    args = parser.parse_args()
    task_name = args.task
    inputs = args.input
    out = args.output
    desc = args.description

    task = TASKS[task_name]()
    # run task
    save_path = out or args.input[0]
    data, save_path = task.process(inputs, save_path=save_path)

    if args.clearml:
        prevs = []
        for inp in inputs:
            vers = get_version_from_path(inp)
            prev = ClearMLDataset.get(vers)
            if prev:
                prevs.append(prev.id)
        
        manager = ClearMLDataset.create(save_path, 
                                        previous=prevs,
                                        tags=[task_name],
                                        description=desc or task_name)
