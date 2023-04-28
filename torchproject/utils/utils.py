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


EXPERIMENTS_DIR = 'dev/experiments'
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

class ModelParamsHandler:
    def __init__(self, model:torch.nn.Module):
        self.model = model
        self.accumulated_grads = []

    def init_params(self, *args, **kwargs):
        """
        Set specific weights initialization
        """
        raise NotImplementedError()

    def get_num_params(self):
        total_n = 0
        for name, p in self.model.named_parameters():
            # print(name, n)
            assert p.requires_grad
            n = p.numel()            
            total_n += n
        return total_n
    
    def validate_grads(self):
        """
        Check if all grads are not NaN. 
        Use it after backward()
        Raises:
            Exception: "Grad of '{LAYER}' is NaN
            Exception: "Grads not defined. Use backward() before"
        """
        for name, param in self.model.named_parameters():
            # print(name)
            try:
                grads = param.grad.data
            except AttributeError:
                msg = "Grads not defined. Use backward() before"
                raise Exception(msg)
            if torch.any(torch.isnan(grads)).item():
                raise Exception(f"Grad for param '{name}' is NaN")
            
    def get_grads(self) -> dict:
        """
        Get gradients summary - tuple(mean, std)
        per each layer
        Returns:
            dict: {'layer.name': ( mean(float), std(float) ) }
        """
        name_grads = dict()
        for name, param in self.named_parameters():
            assert param.requires_grad
            try:
                grads = param.grad.data
            except AttributeError:
                msg = "Grads not defined. Use backward() before"
                raise Exception(msg)
            mean = grads.mean().item()
            std = torch.std(grads).item()
            name_grads[name] = (mean, std)
            
        return name_grads 
    
    def accumulate_grads(self):
        grads = self.get_grads()
        self.accumulated_grads.append(grads)

    def grads2csv(self, csv_path:str):
        """
        Accumulated gradients to file.csv
        |--step--|--layer.name/mean--|--layer.name/std--|
        """
        data = []  # dicts
        for step_grads in self.accumulated_grads.items():
            new_step_grads = dict()
            for name, (mean, std) in step_grads.items():
                new_step_grads[name + '/mean'] = mean
                new_step_grads[name + '/std'] = std
            data.append(new_step_grads)
        
        df = pd.DataFrame.from_dict(data, orient='columns')
        df.to_csv(csv_path, sep=' ', index_label='step')

    def reset(self):
        self.accumulated_grads = []


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
