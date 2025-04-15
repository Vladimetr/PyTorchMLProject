"""
Additional functions
"""
from typing import Union, List
import os
import os.path as osp
from typing import List
from collections import OrderedDict
from tqdm import tqdm
from glob import glob, iglob
import logging
import oyaml
import re
import numpy as np
import torch
import torchaudio
"""

"""
from torchaudio import save as save_audio
try:
    # unneccessary libs
    import librosa
    import matplotlib.pyplot as plt
except ImportError:
    pass


EXPERIMENTS_DIR = '/app/dev/experiments'


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

def load_audio(audio_path:str, normalize=True, 
               resample_rate:int=None) -> tuple:
    """
    Load audio from file
    I checked that
    torchaudio.load(AUDIO, normalize=True)
    is equal to
    librosa.load(AUDIO, dtype=np.float32)
    OR
    torchaudio.load(AUDIO, normalize=True) + Resample
    is close to (1e-7)
    librosa.load(AUDIO) with resample
    Args:
        resample_rate (int, None): resample to target SR
            If None, use source sr
    Returns:
        tuple
          (CH, S): sample
          int: sample rate
    """
    try:
        sample, sr = torchaudio.load(audio_path, normalize=normalize)
    except TypeError:
        sample, sr = torchaudio.load(audio_path, normalization=normalize)
    if resample_rate and resample_rate != sr:
        resample = torchaudio.transforms.Resample(sr, resample_rate)
        sample = resample(sample)
        sr = resample_rate

    return sample, sr
    
def xor(a, b) -> bool:
    """
    either a or b
    only of them must be true
    """
    return int(bool(a)) + int(bool(b)) == 1

def overwrite_hparams(config:dict, hparams:dict):
    """
    Update values of params in config
    with values from hprams
    Args:
        config (dict): parsed config.yaml
        hparams (dict): hierarchical in format
            part:param:value
    NOTE: In dict 1 update only key:value given in dict 2
    """
    for k, v in hparams.items():
        if k not in config:
            raise ValueError(f"Invalid key '{k}' in source dict")
        if not isinstance(v, dict):
            config[k] = v
        else:
            overwrite_hparams(config[k], v)

def parse_unknown_kwargs(kwargs:List[str]) -> dict:
    """
    For example ['--port', '8794', '-p', '693']
    >> {"port": 8794, "p": 693}
    """
    if len(kwargs) % 2 != 0:
        raise ValueError("Key-value pairs only supported")
    out_dict = dict()
    for i in range(0, len(kwargs), 2):
        k, v = kwargs[i], kwargs[i + 1]
        k = k.strip("-")
        if v in ["true", "True"]:
            out_dict[k] = True
            continue
        if v in ["false", "False"]:
            out_dict[k] = False
            continue

        for type in (int, float):
            try:
                v = type(v)
            except:
                continue
            else:
                break
        
        out_dict[k] = v
    return out_dict
        

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


def read_classes(classes:Union[List[str], str]) -> List[str]:
    """
    Read classes from list or from text file
    """
    if isinstance(classes, list):
        # do nothing
        return classes
    
    # read from file
    with open(classes, 'r') as f:
        lines = f.readlines()
    lines = [l.replace("\n", "") for l in lines]
    classes = [l for l in lines if l]
    return classes

def get_sample_dur(data, sr:int) -> float:
    """Get audio duration (in sec) of one sample (row) in dataset
    Args:
        data (pd.DataFrame): ["audio", "start", "end", "label"]
        sr (int): sample rate
    Returns:
        float: sample duration in sec
    """
    dur = list((data["end"] - data["start"]).unique())
    if len(dur) > 1:
        raise Exception(f"Data has different sample dur: {dur}")
    return dur[0] / sr

def get_progress_bar(iter, title:str='', total:int=None):
    bar_fmt = '| {n_fmt}/{total_fmt} {postfix}'
    total = total or len(iter)
    iter = tqdm(iter, desc=title, total=total,
                bar_format='{l_bar}{bar:29}' + bar_fmt)
    return iter


def parse_paths(path:str):
    """
    Scan files according from input file(s)
    Args:
        path (str): in format
            "/path/file1.csv,/path/file*.txt"
        NOTE: sometimes it's better not to load big
        amount of files to RAM using list. 
        Use `parse_paths_iter`
    Returns:
        list: list of files
    """
    paths = []
    for p in path.split(","):
        paths += glob(p, recursive=True)
    return paths
    
def parse_paths_iter(path:str):
    for path in path.split(","):
        for fpath in iglob(path, recursive=True):
            yield fpath


def get_version_from_path(data_path:str, parse=False):
    """
    Extract version info from data path
    Args:
        data_path (str): /path/to/file.v{X}.{Y}-{postfix}
        parse (bool): whether to extract X, Y, postfix into tuple
    Raises:
        ValueError: "Failed to extract version info 
                    from '{data_path}'"
    Returns:
        str: '{X}.{Y}-{postfix}' or
            '{X}.{Y}' or
            '{X}-{postfix}' or
    """
    full_pattern = r'.v\d+([.]\d+)?([-]\w+)?.csv\Z'
    version_pattern = r'v\d+([.]\d+)?([-]\w+)?'
    if not re.search(full_pattern, data_path):
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


def split_silence(audio:Union[torch.Tensor, np.ndarray],
                  sr=16000,
                  top_db=20,
                  frame_len=2048,
                  frame_step=512,
                  save_wav:str=None
                  ) -> np.ndarray:
    """
    Split audio by silence parts using librosa
    Args:
        audio (str): /path/to/audio.wav
            (Tensor, ndarray): (1, S)
        sr (int): sample rate
        top_db (int): hreshold (in decibels) below to consider as silence
        frame_len (int): number of samples per analysis frame
        frame_step (int): number of samples between analysis frames
        save_wav (str, None): /path/to/save/audio.wav
    Returns:
        np.ndarray (M, 2): [[start, end]]
    """
    if isinstance(audio, str):
        audio, _ = load_audio(audio)
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    # (1, S)
    if not audio.shape[0] == 1:
        audio = audio[ :1, :]
    
    intervals = librosa.effects.split(
                    y=audio, 
                    frame_length=frame_len, 
                    hop_length=frame_step,
                    top_db=top_db
                )
    # (M, 2) [[start, end]]

    if save_wav:
        samples = []
        for interval in intervals:
            start, end = interval[0], interval[1]
            samples.append(audio[:, start : end])

        samples = np.concatenate(samples, axis=1)
        samples = torch.from_numpy(samples)
        save_audio(save_wav, samples, sample_rate=sr)

    return intervals


def normalize_confusion_matrix(conf_matrix, by='row'):
    """
    Args:
        conf_matrix (np.ndarray): raw confusion matrix (C, C)
        by (str): Type of normalization ('row', 'col', 'all').
    NOTE: NaN may exist. They will be replaced to 0
    Returns:
        np.ndarray: normalized confusion matrix (C, C)
    """
    if by == 'row':
        normed = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    elif by == 'col':
        normed = conf_matrix.astype('float') / conf_matrix.sum(axis=0)[np.newaxis, :]
    elif by == 'all':
        raise NotImplementedError("yet")
    else:
        msg = "Unknown normalization type. Use 'row', 'col', or 'all'."
        raise ValueError(msg)
    
    # NaN may exist
    normed = np.nan_to_num(normed, nan=0.0)
    assert not np.any(np.isnan(normed))

    normed = np.round(normed, 2)
    
    return normed


def get_next_exprun(dir_:str, tmplt:str, return_num=False):
    n_digits = n_tmplts = 0
    for i in range(2, 4):
        num_tmplt = "\d{" + str(i) + "}"
        try:
            before, after = tuple(tmplt.split(num_tmplt))
            n_digits = i
            n_tmplts += 1
        except:
            continue
    if n_tmplts != 1:
        msg = f"'tmplt' must have only one "\
               "numeration template: '\d{2}' or '\d{3}"
        raise ValueError(msg)
    
    # get all numbers of experiments in given dir
    exps = list(filter(lambda f: re.search(tmplt, f),
                       os.listdir(dir_)))

    max_num = 0
    for exp in exps:
        num = int(exp[len(before) : len(before) + n_digits])
        max_num = max(max_num, num)
    new_num = max_num + 1
    if return_num:
        return new_num
    fmt = "{:0" + str(n_digits) + "d}"
    new_name = before + fmt.format(new_num) + after
    return new_name


def tb_write():
    from torch.utils.tensorboard import SummaryWriter
    import pandas as pd
    writer = SummaryWriter("dev/tb", comment="vova")

    data = pd.read_csv("/mnt/nvme/vovik/noise_classification/dev/grads.csv", sep=" ")
    steps = len(data)


    for i in range(steps):
        row = data.iloc[i].to_dict()
        step = row.pop("step")
        assert int(step) == i
        for name, value in row.items():
            writer.add_scalar(name, value, i)


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    # From https://github.com/fschmid56/EfficientAT/blob/a425fdce92572e602a1d5634799bd9f1f2efa806/helpers/utils.py#L56
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)
    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return wrapper


def get_obj_value(metrics:dict, objective:str) -> float:
    """
    Parse dict with metrics and return objective value
    Args:
        metrics (dict): dict with metrics
        objective (str): "Metrics/train_{name}" 
                        or "Metrics/test_{name}" 
                        or "Loss/train_{name}"
                        or "Loss/test_{name}"
    Raises:
        KeyError: Given objective doesn't exists in metrics dict
    Returns:
        float/int: objective value
    """
    # split title and name
    title, objective = objective.split("/")
    # "train_{metric}" or "test_{metric}"
    assert objective.startswith("train_") or \
            objective.startswith("test_")
    objective = objective.replace("test_", "").replace("train_", "")
    if title == "Loss":
        objective += "Loss"
    value = metrics[objective]
    return value

