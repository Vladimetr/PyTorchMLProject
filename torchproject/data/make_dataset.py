"""
Data pipeline: clean, preprocess, balancing, etc
with different data versions
"""
from os import path as osp
import re
from abc import ABCMeta
from typing import Iterable, Optional, List
import argparse
import pandas as pd
from tqdm import tqdm

COLUMNS = [
    "path",   # abs/path/to/audio.wav
    "shape",  # shape of numpy sample
    "start",  # start of sample part
    "end",    # end of sample part
    "label"   # index of class
]
CSV_SEP = ' '
SR = 8000  # sample rate
VALID_LABELS = [0, 1, 2]


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

    @staticmethod
    def _get_next_version(manifest_path: str) -> str:
        """
        if 'corpuse.txt' -> 'corpuse_v2.txt'
        if 'corpuse_v2.txt' -> 'corpuse_v3.txt'
        """
        assert osp.exists(manifest_path)
        while True:
            _, ext = osp.splitext(manifest_path)
            fbasepath = manifest_path[ :-len(ext)]
            # get version postfix '*.v6'
            postfix_v = re.findall(r'.v\d+', fbasepath)
            if not postfix_v:
                # init first version
                v1 = '.v1'
                fbasepath += v1
                postfix_v = [v1]
            assert len(postfix_v) == 1
            postfix_v = postfix_v[0]

            # '.v5' -> 5
            v = int(postfix_v[2: ])
            # increment version
            v += 1
            # collate new fname
            manifest_path = fbasepath[ :-len(postfix_v)] + f".v{v}" + ext
            if not osp.exists(manifest_path):
                break
        return manifest_path
    
    def _get_progress_bar(self, iter: Iterable) -> Iterable:
        bar_fmt = '| {n_fmt}/{total_fmt} {postfix}'
        iter = tqdm(iter, 
                    desc=self.name, 
                    total=len(iter),
                    bar_format='{l_bar}{bar:29}' + bar_fmt)
        return iter

    def _load_manifest(self, manifest_path: str):
        self.data = pd.read_csv(manifest_path, sep=CSV_SEP)
        self.data.sort_values(by=['path', 'start'], inplace=True)
        self.data_size = len(self.data)

    def _save_manifest(self, manifest_path: str, overwrite = False):
        if osp.exists(manifest_path) and not overwrite:
            manifest_path = self._get_next_version(manifest_path)
        self.data.to_csv(manifest_path, sep=CSV_SEP,
                         header=True, index=False)
        
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
                self.data.loc[i] = [upd_row[k] for k in COLUMNS] 
            else:
                delete_rows.append(i)
        # delete rows
        self.data.drop(delete_rows, inplace=True)

    def process(self, manifest_in: str, 
                manifest_out: Optional[str] = None):
        self._load_manifest(manifest_in)

        self.process_rows()

        manifest_out = manifest_out or manifest_in
        self._save_manifest(manifest_out, overwrite=False)
        

class DropDuplicatesProcessor(DataProcessor):
    """
    Drop rows with same set of params
    ["path", "start", "end"]
    """
    def __init__(self):
        self.name = 'drop-duplicates'

    def process(self, manifest_in: str,
                manifest_out: Optional[str] = None):
        self._load_manifest(manifest_in)

        columns = ["path", "start"]
        self.data.drop_duplicates(subset=columns, inplace=True)

        manifest_out = manifest_out or manifest_in
        self._save_manifest(manifest_out, overwrite=False)
        

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


class JoinManifestsProcessor(DataProcessor):
    def __init__(self):
        self.name = 'join-manifests'

    def process(self, manifests_in: List[str], 
                manifest_out: Optional[str] = None):
        datas = []
        for manifest_in in manifests_in:
            self._load_manifest(manifest_in)
            datas.append(self.data)

        # collate pd.DataFrames
        self.data = pd.concat(datas, ignore_index=True)

        manifest_out = manifest_out or manifest_in
        self._save_manifest(manifest_out, overwrite=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing pipeline')
    parser.add_argument('--task', '-t', type=str, required=True,
                        choices=['drop-duplicates',
                                 'drop-invalid-labels',
                                 'join-manifests',

                                 ],
                        help='task to apply')
    parser.add_argument('--input', '-i', type=str, 
                        nargs='+',
                        required=True,
                        help='path/to/input/manifest.csv')
    parser.add_argument('--output', '-o', type=str, 
                        help='path/to/output/manifest.csv')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    tasks = {
        "drop-duplicates": DropDuplicatesProcessor,
        "drop-invalid-labels": DropInvalidLabelsProcessor,
        "join-manifests": JoinManifestsProcessor,

    }
    task = tasks[args.task]()

    inp = args.input
    if len(inp) == 1:
        inp = inp[0]

    # run task
    task.process(inp, args.output)
