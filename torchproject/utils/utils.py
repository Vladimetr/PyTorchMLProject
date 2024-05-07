"""
Additional functions
"""
import os.path as osp
from typing import List
from collections import OrderedDict
import logging
import argparse
import oyaml
import re
try:
    # unneccessary libs
    from tqdm import tqdm
except ImportError:
    pass


EXPERIMENTS_DIR = 'dev/experiments'

def config_from_yaml(yaml_path:str) -> dict:
    with open(yaml_path) as f:
        config = oyaml.load(f, Loader=oyaml.FullLoader)
    config = OrderedDict(config)
    return config


def dict2yaml(data:dict, yaml_path:str):
    with open(yaml_path, 'w') as f:
        f.write(oyaml.dump(data))


def pprint_dict(d:dict):
    for k, v in d.items():
        print(f"- {k}: {v}")


def update_given_keys(source_dict:dict, update_keys:dict):
    """
    In dict 1 update only key:value given in dict 2
    """
    for k, v in update_keys.items():
        if k not in source_dict:
            raise ValueError(f"Invalid key '{k}' in source dict")
        if not isinstance(v, dict):
            source_dict[k] = v
        else:
            update_given_keys(source_dict[k], v)


def get_logger(name='main', logfile:str=None):
    if name == 'main':
        fmt = "%(asctime)s | %(message)s "
    else:
        fmt = "%(message)s"
    formatter = logging.Formatter(fmt, "%H:%M.%S")
    logger = logging.getLogger(name)
    if logfile:
        handler = logging.FileHandler(logfile, 'a')
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(20)
    return logger


def get_progress_bar(iter, title:str='', total:int=None):
    bar_fmt = '| {n_fmt}/{total_fmt} {postfix}'
    total = total or len(iter)
    iter = tqdm(iter, desc=title, total=total,
                bar_format='{l_bar}{bar:29}' + bar_fmt)
    return iter


def save_features_list(features:List[str], fpath:str):
    with open(fpath, 'w') as f:
        for feat in features:
            f.write(feat + '\n')


def parse_data_path(data_path:str) -> dict:
    """
    Args:
        data_path (str): /path/to/data.v{X}.{Y}-{postfix} or
            /path/to/data.v{X}.{Y} or
            /path/to/data.v{X} or
            /path/to/data.v{X}-{postfix}
    Raises:
        ValueError: "Failed to extract version info 
                    from '{data_path}'"
    Returns:
        dict: {
            'basepath': /path/to/data',
            'major_version': X (int),
            'minor_version': Y (int) or None,
            'postfix': str or None
        }
    """
    ext = osp.splitext(data_path)[1]
    full_pattern = r'.*v\d+([.]\d+)?([-]\w+)?' + ext
    version_pattern = r'v\d+([.]\d+)?([-]\w+)?'
    if not re.fullmatch(full_pattern, data_path):
        raise ValueError("Failed to extract version info from "\
                         f"'{data_path}'")
    match_ = re.search(version_pattern, data_path)
    version_info = match_[0][1: ]  # '{X}.{Y}-{postfix}
    xy = version_info.split('-')[0]  # '{X}.{Y}'
    x = int(xy.split('.')[0])
    try:
        y = xy.split('.')[1]
    except IndexError:
        y = None
    try:
        postfix = version_info.split('-')[1]
    except IndexError:
        postfix = None
    out = {
        "major_version": x,
        "minor_version": y,
        "postfix": postfix
    }
    return out


def get_version_from_path(data_path:str, parse=False):
    """
    Extract version info from data path
    Args:
        data_path (str): /path/to/file.v{X}.{Y}-{postfix}
    Raises:
        ValueError: "Failed to extract version info 
                    from '{data_path}'"
    Returns:
        str: '{X}.{Y}-{postfix}' or
            '{X}.{Y}' or
            '{X}-{postfix}' or
    """
    data_basepath = osp.splitext(data_path)[0]
    full_pattern = r'.*v\d+([.]\d+)?([-]\w+)?'
    version_pattern = r'v\d+([.]\d+)?([-]\w+)?'
    if not re.fullmatch(full_pattern, data_basepath):
        raise ValueError("Failed to extract version info from "\
                         f"'{data_path}'")
    match_ = re.search(version_pattern, data_path)
    version = match_[0][1: ]  
    # '{X}.{Y}-{postfix}
    if not parse:
        return version
    
    xy = version.split('-')[0]
    try:
        postfix = version.split('-')[1]
    except IndexError:
        postfix = None
    x = int(xy.split('.')[0])
    try:
        y = int(xy.split('.')[1])
    except IndexError:
        y = None
    return x, y, postfix 


def get_next_version(data_path: str) -> str:
    """
    if 'corpuse.txt' -> 'corpuse.v2.txt'
    if 'corpuse.v2.txt' -> 'corpuse.v3.txt'
    if 'corpuse.v2.0.txt' -> 'corpuse.v2.1.txt'
    """
    assert osp.exists(data_path)
    ext = osp.splitext(data_path)[1]
    fbasepath = data_path.split('.')[0]
    try:
        x, y, postfix = get_version_from_path(data_path, parse=True)
    except ValueError:
        x, y, postfix = 1, None, None
    while True:
        if y is not None:
            y += 1
        else:
            x += 1

        data_path = fbasepath + f".v{x}"
        if y is not None:
            data_path += f".{y}"
        if postfix is not None:
            data_path += f"-{postfix}"
        data_path += ext

        if not osp.exists(data_path):
            break

    return data_path


if __name__ == '__main__':
    path = 'data.v22-test.csv'
    print(get_version_from_path(path))
    print(get_version_from_path(path, parse=True))
    print()
    print(get_next_version(path))