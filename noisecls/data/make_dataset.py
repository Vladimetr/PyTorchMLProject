"""
Data pipeline: clean, preprocess, balancing, etc
with different data versions
"""
import os
import os.path as osp
from typing import List
import pandas as pd
from torchaudio import load as load_audio
from torchaudio.transforms import Resample
from . import COLUMNS, SR, CSV_SEP, DataProcess
from .transforms import *
from ..import utils
try:
    import ffmpeg
except ImportError:
    pass


DUR = 2.0   # duration for classification


class FormManifest(DataProcess):
    """
    Form manifest from dir with audios
    Args:
        input (str): in format
            "/path/audio1.wav,/path/audios*.csv"
        label (str): label for these audios. See VALID_LABELS
        dur (float): audio duration (sec) for one sample (row) in manifest
        output (str): /path/save/manifest.csv
            If None, {wavs_dir}.csv
    """
    name = "form-manifest"
    def __init__(self, label:str, dur:float=DUR):
        super().__init__()
        self.label = label
        self.dur = dur

    def process(self, input:str, output:str=None):
        """
        input (str): /path/to/wavs/dir/*.wav
        """
        if not (output and output.endswith(".csv")):
            raise ValueError("Output must be *.csv")
    
        data : List[tuple] = []  # [(audio, start, end, label)]
        total_dur = 0.0

        # there may be lrge amount of files, better to use iter
        for wav_path in utils.parse_paths_iter(input):
            sample, sr = load_audio(wav_path)
            if not sr == SR:
                # resample
                sample = Resample(sr, SR)(sample)
            # (*, S)
            slen = sample.shape[1]
            total_dur += slen / sr
            dur_sec = int(self.dur * SR)
            for st in range(0, slen - dur_sec + 1, dur_sec):
                ed = st + dur_sec
                data.append(
                    (wav_path, st, ed, self.label)
                )

        # form CSV
        df = pd.DataFrame(data, columns=COLUMNS)
        df.to_csv(output, sep=CSV_SEP, index=False, header=True)
        self.data = df
        self.out_path = output
        print(f"Total dur for {input}: {total_dur} sec")


class ConvertAudios(DataProcess):
    """
    Convert audios in dir using FFMPEG and save to another dir
    NOTE: name of audio will be the same
    """
    name = "convert"
    def __init__(self, overwrite=True):
        super().__init__()
        self.overwrite = overwrite
        self.params = {
            'ar': str(SR),
            'acodec': 'pcm_s16le',
            'f': 'wav',
            'loglevel': 'error',
        }

    def process(self, input:str, output:str=None):
        """
        Args:
            input (str): in format
                "/path/audio1.wav,/path/audios*.csv"
            output (str): /path/to/save/converted/audios/
        """
        if not (output and osp.exists(output) and input != output):
            raise ValueError("new output dir must exist")

        # there may be lrge amount of files, better to use iter
        for wav_path in utils.parse_paths_iter(input):
            wav_name = osp.split(wav_path)[1]
            path_out = osp.join(output, wav_name)

            # prevent upsampling
            info = ffmpeg.probe(wav_path)
            sr = int(info['streams'][0]['sample_rate'])
            assert sr >= SR, wav_path

            if osp.exists(path_out):
                if not self.overwrite:
                    raise ValueError(f"{wav_name} already exists")
                os.remove(path_out)  # it's like over-write
            stream = ffmpeg.input(wav_path)
            convert = ffmpeg.output(stream, path_out, **self.params)
            ffmpeg.run(convert)


class GetClasses(DataProcess):
    """Get set of classes from manifest
    """
    name = "get-classes"
    def __init__(self):
        super().__init__()

    def process(self, input:str, output:str=None):
        """
        Args:
        input (str): /path/to/input/manifest.csv
        output (str, optional): /path/to/save/classes.txt
            If None, /path/to/input/manifest-classes.txt
        """
        self._load_data(input)
        classes = self.data['label'].unique()
        if not output:
            output = input.replace(".csv", "-classes.txt")
        
        with open(output, 'w') as f:
            for cls_ in classes:
                f.write(str(cls_) + "\n")
        print("Saved", output)   

    
class SplitTrainTest(DataProcess):
    name = "split"
    def __init__(self, test_ratio:float=0.2):
        super().__init__()
        self.test_ratio = test_ratio

    def process(self, input:str, output:str=None):
        # read input data
        self._load_data(input)
        n = len(self.data)
        test_n = int(n * self.test_ratio)
        self.data = self.data.sample(frac=1)  # shuffle
        # define output path
        output = output or input
        if osp.isdir(output):
            name = osp.split(input)[1]
            output = osp.join(output, name)
        test_path = output.replace(".csv", "-test.csv")
        train_path = output.replace(".csv", "-train.csv")
        # split
        test_data = self.data[ :test_n]
        test_data.to_csv(test_path, sep=CSV_SEP,
                        header=True, index=False)
        train_data = self.data[test_n: ]
        train_data.to_csv(train_path, sep=CSV_SEP,
                        header=True, index=False)
        print("Splited:")
        print(f"{int(self.test_ratio * 100)}%", test_path)
        print(f"{int((1 - self.test_ratio) * 100)}%", train_path)
     

def get_valid_tasks():
    classes = list(filter(lambda x: isinstance(x, type(DataProcess)),
                          globals().values()))
    tasks = {c.name : c for c in classes}
    return tasks

TASKS = get_valid_tasks()

def main(args:dict):
    # read main args
    task = args.pop("task")
    inp = args.pop("input")
    out = args.pop("output")
    descr = args.pop("description")
    vers = args.pop("version")
    tags = args.pop("tags")
    manager = args.pop("clearml")
    # other args are specific for task

    try:
        processor : DataProcess = TASKS[task]
    except KeyError:
        raise ValueError(f"Invalid task name '{task}'")
    # init
    processor = processor(**args)
    # process
    processor.process(inp, out or inp)
    if manager:
        tags = tags.split(",") if tags else None
        processor.add_manager_info(tags=tags, description=descr, version=vers)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data processing functions')
    parser.add_argument('--task', '-t', type=str,
                        choices=list(TASKS.keys()),
                        help='task to apply')
    parser.add_argument('--input', '-i', type=str, 
                        required=True,
                        help='path/to/input/data.csv. Use quotes for regex!')
    parser.add_argument('--output', '-o', type=str, 
                        help='path/to/output/data.csv')
    parser.add_argument('--clearml', action='store_true', 
                        default=False, 
                        help='whether to use ClearML')
    parser.add_argument('--description', '-d', type=str, default=None, 
                        help='Description for ClearML')
    parser.add_argument('--version', '-v', type=str, default=None, 
                        help='Version in ClearML. In format "2.0-postfix"')
    parser.add_argument('--tags', type=str, nargs='+', default=None, 
                        help='Tags in ClearML')
    args, unknown = parser.parse_known_args()
    kwargs = utils.parse_unknown_kwargs(unknown)
    # additional algorithm params, for ex. filter params

    main({**vars(args), **kwargs})
