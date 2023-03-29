"""
вспомогательные функции и/или классы
например Конвертация, Трансформация, подсчет ошибок и т п
"""
import os
from itertools import product
import pandas as pd
import oyaml
from collections import OrderedDict
import logging
import torch


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



def grid_search(csv_path=None):
    """
    Перебор всех возможных комбинаций гиперпараметров
    и сохранение в файл .csv
    :param csv_path: куда сохранять комбинации параметров
                    if None, just print
    :return list of dicts (всех возможных комбинаций)
    """
    # variable
    params = {
        'restore': None,
        'n_classes' : 4,
        'lr': [1e-3, 1e-4],
        'lr_reduce_ep' : 2,     # reduce learning rate every 2 epochs
        'weight_decay' : 1e-5,   # RegLoss coefficient
        'grad_norm': [None, 1., 5.],  # max norm of gradient (see clip_grad_norm). None - without clipping
        'epochs': 15,
        'opt': ['Adam', 'sgd'],
        'batch_size': 500,
        'steps': None,            # interrupt epoch at steps. None - full epoch
        'logdir': 'logdir'        # logs for ALL models
        # other params
    }

    ls_params = []
    for _, v in params.items():
        ls_params.append(
            v if isinstance(v, list) else [v])

    combs = list(product(*ls_params))

    keys = list(params.keys())
    ls_params = []
    for comb in combs:
        dic = {}
        for i in range(len(comb)):
            dic[keys[i]] = comb[i]

        ls_params.append(dic)

    if csv_path:
        # save params to csv
        pass
    else:
        for i, params in ls_params:
            print('{}) {}'.format(i, params))



def split_train_test(data_dir, dist_dir, test_ratio=0.2):
    """
    разбивает выборку на train/test
    :param data_dir:
    :param dist_dir:
    :param test_ratio: доля тестовой части
    :return: dist_dir/train.csv and dist_dir/test.csv
    """
    data_csv = os.path.join(data_dir, 'data.csv')
    df = pd.read_csv(data_csv, sep=',', header=None)
    df = shuffle(df)

    assert 0 < test_ratio < 1.
    train_csv = os.path.join(dist_dir, 'data_train.csv')
    test_csv = os.path.join(dist_dir, 'data_test.csv')

    df_train = df[:int((1-test_ratio) * len(df))]
    df_train.to_csv(train_csv, sep=',', header=None, index=False)

    df_test = df[int((1-test_ratio) * len(df)):]
    df_test.to_csv(test_csv, sep=',', header=None, index=False)

    return train_csv, test_csv


def change_logger(logger, new_logfile, format='%(message)s'):
    fileh = logging.FileHandler(new_logfile, 'a')
    formatter = logging.Formatter(format)
    fileh.setFormatter(formatter)

    log = logger.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)      # set the new handler

def confusion_matrix(y_pred, y_true):
    """
    Confusion matrix
    :param y_pred: torch.Tensor with shape (batch_sz, n_classes)  (0, 1)
    :param y_true: torch.Tensor with shape (batch_sz, ) - номер класса
                                                          0..n_classes-1
    
    :return matrx: torch.Tensor [0..1] with shape (n_classes, n_classes)
    """
    assert y_pred.ndim == 2 and y_true.ndim == 1
    assert y_pred.size()[0] == y_true.size()[0]
    
    bs, n_classes = y_pred.size()
    
    y_pred = y_pred.argmax(dim=1)
    # y_pred (bs, ) - номер класса 0..n_classes-1
    
    data = torch.cat((y_pred.view(bs, 1), y_true.view(bs, 1)), dim=1)
    # (bs, 2) column 'pred' and column 'true'
    
    # квадратная матрица
    matrx = torch.zeros(n_classes, n_classes)
    
    for i in range(bs):
        matrx[tuple(data[i])] += 1
        
    # normalize
    for i in range(n_classes):
        matrx[:, i] = matrx[:, i] / torch.sum(matrx[:, i])

    return matrx


def plot_confusion_matrix(conf_matrx, classes, save_name='conf_matrix.png'):
    """
    :param conf_matrix: normalized ndarray with shape (n_classes, n_classes)
    :param classes: ls of names ['noise', 'speech', 'music']
    :param save_name: path to save fig .png
    """
    # check normalized
    if np.any(conf_matrx > 1.0) or np.any(conf_matrx < 0):
        raise Exception('Confusion Matrix must be normalized (0, 1)')
    
    fig, ax = plt.subplots(figsize=(10,7))
    im = ax.imshow(conf_matrx, cmap='Blues')

    # We want to show all ticks ...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes, fontsize=20)
    ax.set_yticklabels(classes, fontsize=20)

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            c = 'w' if conf_matrx[i, j] > 0.5 else 'k'
            value = '{:.2f}'.format(conf_matrx[i, j])
            text = ax.text(j, i, value,
                           ha="center", va="center", color=c, fontsize=30)

    ax.set_title("Confusion marix", fontsize=20)

    fig.colorbar(im, shrink=1.0, extend='neither',
                drawedges=False, orientation='vertical')
    
    # savefig
    fig.savefig(save_name)
    plt.close()
    
    
class Metrics():
    def __init__(self, acc=True, another_metrics=False):
        self.acc = 0 if acc else None
        self.another_metrics = 0 if another_metrics else None
        # ...
        self.n = 0

    def __call__(self, probs, target):
        metrics = {}
        if self.acc is not None:
            acc = formula
            metrics['acc'] = acc
            self.acc += 0

        if self.another_metrics is not None:
            anoher_metrics = formula
            self.another_metrics += another_metrics
            metrics['another_metrics'] = another_metrics
            
        self.n += 1
        return metrics

    def get_avg(self):
        metrics = {}
        
        if not self.n:
            return None
        
        if self.acc is not None:
            metrics['acc'] = self.acc / self.n
            self.acc = 0
        if self.another_metrics is not None:
            metrics['another_metrics'] = self.another_metrics / self.n
            self.another_metrics = 0

        self.n = 0
        return metrics






