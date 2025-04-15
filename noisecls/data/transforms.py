"""
Classes for data transformation
Data transform refers to one (except JoinManifests) 
manifest change only, audios stay unchanged
NOTE: transform take manifest(s) as input - parent version(s)
and produce output - next version
"""
from abc import ABCMeta
from typing import Optional, List, Union
import pandas as pd
import json
from . import CSV_SEP, COLUMNS, SR, DataProcess
from .. import utils

data_type = Union[str, pd.DataFrame]


class DataTransform(DataProcess, metaclass=ABCMeta):
    """
    Base data transformator
    NOTE: if you set kwargs key and value must be string
    for accessable from CLI so it must be able to parse 
    values from string.
    For example to get list of string
    parse it from line like "1,2,3" -> [1, 2, 3]
    """
    name = "some-transform"
    def __init__(self, **kwargs):
        """
        **kwargs: additional params of processing
        """
        self.data : pd.DataFrame = None
        self.meta : dict = None
        self.in_paths : List[str] = []  # [input manifest.csv]
        # it can be multible for specific tasks like join
        self.out_path : str = None  # output manifest.csv
        self.kwargs = kwargs  # additional params for processing
        # don't forget super().__init__(**kwargs) for child classes

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

    def process(self, data_in: data_type,
                save_path: Optional[str]=None):
        """
        Main method for running process
        Args:
            data_in: 
                1) (str) /path/to/single/manifest.csv
                2) pd.Dataframe
            save_path (str, None): /path/to/manifest.csv
                If path exists, use next version (v2->v3)
                If None, no save
            NOTE: if you need to save as next version (v2->v3)
            set save_path equal to data_in
        Returns:
            tuple
              pd.DataFrame: processed data
              str: /path/to/saved/data.csv or None
        """
        print(f"Process task {self.name}...")
        if isinstance(data_in, str):
            print(f" -input: {data_in}")
            datas_in = utils.parse_paths(data_in)
            self.in_paths = datas_in
            assert len(datas_in) == 1, \
                "this task requires one manifest as input" 
            self._load_data(datas_in[0])
        elif isinstance(data_in, pd.DataFrame):
            print(f" -input: DataFrame")
            self.data = data_in.copy(deep=True)
            self.data_size = len(self.data)

        self.process_rows()

        if save_path:
            save_path = self._save_data(save_path, overwrite=False)
            self.out_path = save_path
            print(f" -output: {save_path}")

        return self.data, save_path
    

class NoProcess(DataTransform):
    """
    No data change, just saving to ClearML
    is available
    """
    name = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_in: data_type,
                save_path: Optional[str]=None):
        if not isinstance(data_in, str):
            raise ValueError("Input must be str for saving to ClearML")
        datas_in = utils.parse_paths(data_in)
        if len(datas_in) > 1:
            raise ValueError("Input must me single filepath "\
                             "for saving to ClearML")
        self.out_path = datas_in[0]
        self._load_data(datas_in[0])


class DropDuplicates(DataTransform):
    """
    Drop duplucated rows
    """
    name = 'drop-duplicates'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_rows(self):
        self.data.drop_duplicates(inplace=True)
        

class JoinManifests(DataTransform):
    name = 'join-manifests'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_in:str,
                save_path: Optional[str]=None):
        """
        data_in (str): several paths in format
            "/path/manifest1.csv,/path/manifest*.csv"
        """
        print(f"Process task {self.name}...")
        print(f" -input: {data_in}")
        datas_in = utils.parse_paths(data_in)
        self.in_paths = datas_in
        if save_path == data_in:
            save_path = datas_in[0]

        datas = []
        sample_dur = None
        for data in datas_in:
            data = pd.read_csv(data, sep=CSV_SEP)

            # check all datas have same SR
            dur = utils.get_sample_dur(data, SR)
            if sample_dur:
                assert dur == sample_dur
            else:
                sample_dur = dur

            datas.append(data)
        
        # concat
        self.data = pd.concat(datas, ignore_index=True)
        self.data_size = len(self.data)

        # save
        if save_path:
            save_path = self._save_data(save_path, overwrite=False)
            self.out_path = save_path
            print(f" -output: {save_path}")

        return self.data, save_path
    

class FilterClasses(DataTransform):
    name = 'drop-classes'
    def __init__(self, 
                 drop_classes:Union[List[str], str]=None,
                 remain_classes:Union[List[str], str]=None):
        super().__init__(
            drop_classes=drop_classes,
            remain_classes=remain_classes
        )
        self.drop_classes = utils.read_classes(drop_classes) \
                            if drop_classes else []
        self.remain_classes = utils.read_classes(remain_classes) \
                            if remain_classes else []
        if not utils.xor(self.drop_classes, self.remain_classes):
            raise ValueError("Define either drop_classes or remain_classes")

    def process_row(self, row: dict) -> dict:
        class_name = row["label"]
        if self.drop_classes:
            if class_name in self.drop_classes:
                return  # skip row
            return row

        # remain classes    
        if class_name in self.remain_classes:
            return row
        return  # skip row
    

class JoinClasses(DataTransform):
    name = 'join-classes'
    def __init__(self, join_classes:str):
        """
        join_classes (str): /path/to/file.json
        {
            "joined_class": [
                "src_class_1",
                "src_class_2",
                ...
            ]
        }
        """
        with open(join_classes, 'rb') as f:
            self.join_classes : dict = json.load(f)  # dict
        super().__init__(join_classes_json=join_classes)

    def process_row(self, row: dict) -> dict:
        class_name = row["label"]

        for new_class, joined_classes in self.join_classes.items():
            if class_name in joined_classes:
                row["label"] = new_class
                return row
            
        # no change
        return row
    
    def add_manager_info(self, 
                         tags:List[str]=None, 
                         description:str=None,
                         version:str=None):
        super().add_manager_info(tags, description, version, commit=False)
        # + add json
        json_path = self.kwargs["join_classes_json"]
        self.manager.upload_file(json_path, "join-classes.json")
        self.manager.commit()


