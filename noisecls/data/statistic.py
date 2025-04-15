from typing import List, Union
import os
import os.path as osp
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchaudio import save as save_audio
from torchaudio import load as load_audio
from . import SR, CSV_SEP
from ..import utils
from ..utils.manager import ClearMLDataset

HIST_RANGE = (0, 20)
HIST_STEP = 20
TMP_DIR = "data/examples"  # where to save tmp audio


def class_distr(data, sample_time:float=None,
                manager:ClearMLDataset=None) -> dict:
    """
    Total count (or time) for each class
    Args:
        data (pd.DataFrame): ["audio", "start", "end", "label"]
        sample_time (float): duration in sec of one sample (row)
    Returns:
        dict: {"class": count of data samples (rows)}
    """
    distr = dict()
    labels = list(data['label'].unique())
    counts = []
    for label in labels:
        count = len(data[data["label"] == label])
        count = count if not sample_time else sample_time * count
        counts.append(count)
        distr[label] = count

    if manager:
        hist = np.array(counts, dtype=np.uint16)
        manager.add_bar(hist, name="classes", 
                        ytitle="count", xtitle="classes", 
                        labels=labels)
        
    return distr

def audiotime_hist(data, manager:ClearMLDataset=None) -> np.ndarray:
    # sort data by audio name
    data = data.sort_values(by=['audio'])
    prev_audio = None
    n = len(data)
    durs = []
    for i in tqdm(range(n), desc='Audio time'):
        audio = data.iloc[i][0]
        if i+1 == n or (prev_audio and audio != prev_audio):
            # save previous
            # load audio
            sample, sr = utils.load_audio(audio)
            # (*, S)
            assert SR == sr, audio
            slen = sample.shape[1]
            dur = slen / SR
            durs.append(dur)
            
        prev_audio = audio

    if manager:
        bins = list(range(HIST_RANGE[0], HIST_RANGE[1], HIST_STEP)) + [1000]
        manager.add_histogram(durs, bins=bins, name="audio duration", 
                              ytitle="count", xtitle="sec")
    
    return durs


def save_examples(data, n:int=1, 
                  classes:Union[int, List[str]]=None, 
                  output:str=None,
                  manager:ClearMLDataset=None):
    """
    Save some random samples from dataset
    Args:
        data (str, pd.DataFrame): /path/to/manifest.csv or DataFrame
        n (int): num of examples per each given class
        classes (int, list[str]): If list[str], specific classes
                                  If int, number of random classes
                                  If None, all classes
        output (str, None): /path/to/save/dir. If None, 
                            without saving to dir
        manager (ClearMLDataset, None): If None, don't save to ClearML
    """
    if not output and manager:
        # save to tmp dir
        output = TMP_DIR
    os.makedirs(output, exist_ok=True)

    if isinstance(data, str):
        # read csv
        data = pd.read_csv(data, sep=CSV_SEP)

    if isinstance(classes, int):
        # n random classes
        n_classes = classes
        classes = data["label"].unique()
        shuffle(classes)
        classes = classes[ :n_classes]
    elif not classes:
        # all classes
        classes = data["label"].unique()

    for cls_ in classes:
        cls_rows = data.loc[data["label"] == cls_]
        samples = cls_rows.sample(n=n)
        for i in range(len(samples)):
            audio, st, ed, label = samples.iloc[i]
            assert label == cls_
            sample, sr = load_audio(audio)  # (CH, S)
            assert sr == SR and sample.shape[0] == 1, audio
            sample = sample[:, st : ed]  # cut
            name = f"{cls_}_{i+1}.wav"
            if output:
                # save audio to dir
                wav_path = osp.join(output, name)
                save_audio(wav_path, sample, sample_rate=SR)
            if manager:
                manager.add_example(wav_path, name=name)
                manager.add_tags(["examples"])


TASKS = {
    "class": class_distr,
    "audio": audiotime_hist,
    "examples": save_examples,

}

def main(data:Union[str, pd.DataFrame], 
         tasks:List[str]=list(TASKS.keys()), 
         manager:bool=False,
         **kwargs):
    """
    Compute given statistics
    Args:
        data (str, pd.DataFrame): /path/to/manifest.csv or DataFrame
        tasks (List[str]): list of statistic tasks. See TASKS
        manager (ClearMLDataset, bool): True/False whether to use manager
            ClearMLDataset - already inited manager
    """
    if manager:
        # load manager for existed data version
        if not isinstance(data, str):
            raise ValueError("For loading ClearML input data must be str")
        manager = ClearMLDataset.get_by_data(data)
        if not manager:
            raise ValueError(f"Dataset '{args.input}' must "\
                              "exist in ClearML")
    else:
        manager = None
    
    if isinstance(data, str):
        data = pd.read_csv(args.input, sep=CSV_SEP)

    for task in tasks:
        TASKS[task](data=data, manager=manager, **kwargs)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data processing functions')
    parser.add_argument('--tasks', '-t', type=str,
                        nargs="+",
                        default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                       help='type of statistic')
    parser.add_argument('--input', '-i', type=str, 
                        required=True,
                        help='path/to/input/data.csv')
    parser.add_argument('--clearml', action='store_true', 
                        default=False, 
                        help='whether to use ClearML')
    args, unknown = parser.parse_known_args()
    kwargs = utils.parse_unknown_kwargs(unknown)

    main(args.input, args.tasks, args.clearml, **kwargs)
