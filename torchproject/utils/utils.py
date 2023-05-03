"""
вспомогательные функции и/или классы
например Конвертация, Трансформация, подсчет ошибок и т п
"""
import os
import os.path as osp
import pandas as pd
import oyaml
from collections import OrderedDict
import logging
import torch
import argparse
try:
    # unneccessary libs
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
except ImportError:
    pass


EXPERIMENTS_DIR = '/mnt/nvme/vovik/tutorial/torch_project/dev/experiments'
TB_LOGS_DIR = 'dev/tensorboard/logs'


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
    fmt = "%(message)s "
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


def read_metrics_from_csv(csv_path:str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=' ')
    columns = df.columns.tolist()
    drop_columns = ['epoch', 'step']
    for col in columns:
        if col in drop_columns or col.startswith("Unnamed"):
            df = df.drop(columns=[col])
    return df


def tensorboard_log(exp_run_path:str):
    """
    Parse metrics in .csv files
    and log them in Tensorboard
    Args:
        exp_path (str): For ex. 
            'augmentation/train/001/'
            'dev/experiments/augmentation/test/002-1/'
    """
    def write_from_csv(csv_path:str, tb_writer:SummaryWriter, title=None):
        df_metrics = read_metrics_from_csv(csv_path)

        bar_format = '{l_bar}{bar:29}{n_fmt}/{total_fmt}'
        progress_bar = tqdm(range(len(df_metrics)), desc=title, 
                            bar_format=bar_format)
        for step in progress_bar:
            metrics = df_metrics.iloc[step].to_dict()
            for name, value in metrics.items():
                if name == 'loss':
                    name = 'Loss/CrossEntropy'
                else:
                    name = 'Metrics/' + name
                tb_writer.add_scalar(name, value, step)

    exp_dir = exp_run_path.split(EXPERIMENTS_DIR)[-1]
    if exp_dir.startswith('/'):
        exp_dir = exp_dir[1: ]
    # Train metrics
    train_csv = osp.join(exp_run_path, 'train.csv')
    if osp.exists(train_csv):
        log_dir = osp.join(TB_LOGS_DIR, exp_dir.replace('test', 'train'))
        print(f"TB Logs to: '{log_dir}'")
        writer = SummaryWriter(log_dir=log_dir)
        write_from_csv(train_csv, writer, 'train')
        writer.close()
    # Test metrics
    test_csv = osp.join(exp_run_path, 'test.csv')
    if osp.exists(test_csv):
        log_dir = osp.join(TB_LOGS_DIR, exp_dir.replace('train', 'test'))
        print(f"TB Logs to: '{log_dir}'")
        writer = SummaryWriter(log_dir=log_dir)
        write_from_csv(test_csv, writer, 'test')
        writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--run', '-r', type=str, 
                        required=True,
                        help='For ex. augmentation/train/001/')
    args = parser.parse_args()

    tensorboard_log(args.run)
