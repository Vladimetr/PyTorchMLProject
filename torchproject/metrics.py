from collections import OrderedDict
from typing import List
from abc import ABCMeta, abstractmethod
import torch
from  torch import Tensor

class Loss(metaclass=ABCMeta):
    """ Abstract Loss """
    @abstractmethod
    def __init__(self, device='cpu', *args, **kwargs):
        """
        Base parameters
        n_classes (int): number of classes
        device (str): 'cpu' or 'cuda'
        """
        pass

    @abstractmethod
    def __call__(self, pred:Tensor, targ:Tensor) -> dict:
        """
        B - batch size
        C - n classes
        Args:
            pred (B, C): predicted logits
            targ (B, C)): target one hot
        Returns:
            tuple:
              loss: object with method backward()
              dict: {'<NameLoss>': float}
        NOTE: loss object for backward is only one. 
        But dict can contain multiple key:values
        NOTE: <NameLoss> must end with '*Loss'
        """
        pass


class CrossEntropyLoss(Loss):
    def __init__(self, device='cpu', weights=None):
        if isinstance(weights, list):
            weights = Tensor(weights)
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.loss = self.loss.to(device)

    def __call__(self, pred: Tensor, targ: Tensor) -> tuple:
        loss = self.loss(pred, targ)
        ce_value = loss.item()  # float
        loss_values = {
            "CrossEntropyLoss": ce_value,
        }
        return loss, loss_values


def init_loss(name:str, params:dict, device='cpu') -> Loss:
    if name == 'cross_entropy':
        loss = CrossEntropyLoss(device=device, **params)
        
    # another loss
    else:
        raise ValueError(f"Invalid loss '{name}'")
    return loss



class BinClassificationMetrics:
    def __init__(self, compute_metrics:list, logger=None,
                 n_classes:int=2, pos_classes:List[int]=None,
                 class_names:List[str]=None,
                 epoch=False, step=False):
        """
        compute_metrics (list[str]): order of metrics to compute
        n_classes (int): number of classes - C.
            In case of more than 2, it will be converted to binary
        pos_classes (list[int], None): list of indexes 
            of positive classes. For converting multiclass 
            to binary. If None, positive class is 1
        class_names (list[str]): C-list of class names
            If None, classes ['0', '1', ..., C-1]
        epoch (bool): whether to log epoch
        step (bool): whether to log step
        """
        self.compute_metrics = compute_metrics
        # All of these metrics are computed based on conf matrix
        self.metrics_funcs = {
            "TP": self.tp,
            "FN": self.fn,
            "TN": self.tn,
            "FP": self.fp,
            "Acc": self.accuracy,
            "Recall": self.recall,  # TPR
            "Precision": self.precison,

        }
        log_items = list(compute_metrics)
        if step:
            log_items = ['step'] + log_items
        if epoch:
            log_items = ['epoch'] + log_items
        self.log_items = log_items
        if logger:
            # set title
            logger.info(' '.join(log_items))
        self.n_classes = n_classes
        self.pos_classes = pos_classes or [1]
        self.class_names = class_names or \
                           [str(i) for i in range(n_classes)]
        self.logger = logger
        self.epoch = epoch
        self.step = step
        self.reset_summary()

    def conf_matrix(self, pred:Tensor, targ:Tensor,
                        precomputed:dict=None) -> Tensor:
        """
        C - n_classes
             0    2
        0 |    |    |
        2 |    |    |
        columns: actual
        rows: predicted
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            Tensor (C, C)
        """
        conf_matrix = torch.zeros(self.n_classes, self.n_classes,
                                  dtype=torch.int)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                conf_matrix[i, j] = torch.sum(torch.logical_and(
                                        pred == i, targ == j
                ))
        return conf_matrix

    def bin_conf_matrix(self, pred:Tensor, targ:Tensor,
                            precomputed:dict=None) -> Tensor:
        """
             0    1
        0 | TN | FN |
        1 | FP | TP |
        columns: actual
        rows: predicted
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            Tensor (2, 2)
        """
        if not precomputed:
            precomputed = dict()
        tn = precomputed.get("TN", self.tn(pred, targ))
        fn = precomputed.get("FN", self.fn(pred, targ))
        fp = precomputed.get("FP", self.fp(pred, targ))
        tp = precomputed.get("TP", self.tp(pred, targ))
        conf_matrix = torch.tensor(
            [[tn, fn],
             [fp, tp]],
            dtype=torch.int
        )
        return conf_matrix
    
    def conf_matrix_to_binary(self, conf_matrix:Tensor) -> Tensor:
        """
        Converting multiclass confusion matrix to binary matrix
        according to positive classes
        Args:
            conf_matrix (tensor (C, C)): multiclass conf matrix
        Returns:
            tensor (2, 2): binary conf matrix
        """
        fn = fp = tp = 0
        for i in range(self.n_classes):
            for j in range(self.n_classes): 
                if i in self.pos_classes and j in self.pos_classes:
                    tp += conf_matrix[i, j]
                if i in self.pos_classes and j not in self.pos_classes:
                    fp += conf_matrix[i, j]
                if i not in self.pos_classes and j in self.pos_classes:
                    fn += conf_matrix[i, j]
        tn = conf_matrix.sum().item() - tp - fn - fp
        bin_conf_matrix = torch.tensor(
            [[tn, fn],
             [fp, tp]],
            dtype=torch.int
        )
        return bin_conf_matrix

    @staticmethod
    def tp(pred:Tensor, targ:Tensor,
           precomputed:dict=None) -> int:
        """
        True positive
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            int: true positive
        """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            tp = torch.sum(torch.logical_and(pred == 1, targ == 1)).item()
        else:
            tp = conf_matrix[1, 1].item()
        return tp
    
    @staticmethod
    def tn(pred:Tensor, targ:Tensor,
           precomputed:dict=None) -> int:
        """
        True negative
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            int: true negative
        """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            tn = torch.sum(torch.logical_and(pred == 0, targ == 0)).item()
        else:
            tn = conf_matrix[0, 0].item()
        return tn
    
    @staticmethod
    def fp(pred:Tensor, targ:Tensor,
           precomputed:dict=None) -> int:
        """
        False positive
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            int: false positive
        """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            fp = torch.sum(torch.logical_and(pred == 1, targ == 0)).item()
        else:
            fp = conf_matrix[1, 0].item()
        return fp
    
    @staticmethod
    def fn(pred:Tensor, targ:Tensor,
           precomputed:dict=None) -> int:
        """
        False positive
        Args:
            pred (B, ): indexes of pred classes (pos=1)
            targ (B, ): indexes of targ classes (pos=1)
        Returns:
            int: false negative
        """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            fn = torch.sum(torch.logical_and(pred == 0, targ == 1)).item()
        else:
            fn = conf_matrix[0, 1].item()
        return fn
    
    def accuracy(self, pred:Tensor, targ:Tensor, 
                 precomputed:dict=None) -> float:
        """ (TP + TN) / Total """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.bin_conf_matrix(pred, targ)
        tn = conf_matrix[0, 0].item()
        tp = conf_matrix[1, 1].item()
        s = conf_matrix.sum().item()
        try:
            acc = (tn + tp) / s
        except ZeroDivisionError:
            return -1
        return acc
    
    def precison(self, pred:Tensor, targ:Tensor, 
                 precomputed:dict=None) -> float:
        """ TPR = TP / (TP + FP) """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.bin_conf_matrix(pred, targ)
        tp = conf_matrix[1, 1].item()
        fp = conf_matrix[1, 0].item()
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            return -1
        return prec
    
    def recall(self, pred:Tensor, targ:Tensor, 
               precomputed:dict=None) -> float:
        """ TPR = TP / (TP + FN) """
        try:
            conf_matrix = precomputed["bin_conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.bin_conf_matrix(pred, targ)
        tp = conf_matrix[1, 1].item()
        fn = conf_matrix[0, 1].item()
        try:
            rec = tp / (tp + fn)
        except ZeroDivisionError:
            return -1
        return rec

    def compute(self, probs:Tensor, targ:Tensor,
                accumulate=False) -> dict:
        """
        B - batch size
        C - n classes
        Args:
            probs (B, C): probs for each class
            targ (B, )): target indexes class
        Returns:
            dict: dict with metrics
        """

        if probs.shape[0] != targ.shape[0]:
            raise ValueError("Mismatch probs and targ shapes")
        if probs.shape[1] != self.n_classes:
            raise ValueError(f"Invalid number of classes {probs.shape[1]}")
        # class indexes with max prob
        pred = torch.max(probs, dim=1)[1]  # (B, )
        
        metrics = metrics = OrderedDict()
        if self.n_classes > 2:
            conf_matrix = self.conf_matrix(pred=pred, targ=targ)
            metrics["conf_matrix"] = conf_matrix
            bin_conf_matrix = self.conf_matrix_to_binary(conf_matrix)
        else:
            bin_conf_matrix = self.conf_matrix(pred, targ)
        
        metrics["bin_conf_matrix"] = bin_conf_matrix
        for name in self.compute_metrics:
            try:
                func = self.metrics_funcs[name]
                metrics[name] = func(pred=None, targ=None,
                                     precomputed=metrics)
            except KeyError:
                continue

        if accumulate:
            self.sum_bin_conf_matrix += metrics["bin_conf_matrix"]
            if self.n_classes > 2:
                self.sum_conf_matrix += metrics["conf_matrix"]
        # metrics.pop("conf_matrix")
        return metrics
    
    def add_summary(self, metrics:dict):
        """
        Add custom metrics (that are out of conf matrix)
        to summary metrics
        """
        for k, v in metrics.items():
            try:
                self.sum_metrics[k].append(v)
            except KeyError:
                self.sum_metrics[k] = [v]  # new value in list

    def summary(self) -> dict:
        """ Get summary of accumulated metrics """
        metrics = OrderedDict({
            "bin_conf_matrix": self.sum_bin_conf_matrix,
            "conf_matrix": self.sum_conf_matrix,
        })
        for name in self.compute_metrics:
            try:
                func = self.metrics_funcs[name]
                metrics[name] = func(pred=None, targ=None,
                                     precomputed=metrics)
            except KeyError:
                continue 
        # additional metrics (out of conf matrix)
        for k, v in self.sum_metrics.items():
            metrics[k] = sum(v) / len(v)
        return metrics
    
    def reset_summary(self):
        self.sum_conf_matrix = torch.zeros(self.n_classes, self.n_classes,
                                           dtype=torch.int)
        self.sum_bin_conf_matrix = torch.zeros(2, 2,
                                           dtype=torch.int)
        # additional metrics (out of conf matrix)
        self.sum_metrics = dict()  # {'name': list[values]}

    def log_metrics(self, metrics:dict, epoch:int=None, step:int=None):
        items = dict(metrics)
        if epoch:
            items['epoch'] = epoch
        if step:
            items['step'] = step

        if self.logger:
            # log to file in .csv format
            log_line = ' '.join(str(items[col]) for col in self.log_items)
            self.logger.info(log_line)
        else:
            # print to stdout
            log_items = []  # list[str]
            for name in self.log_items:
                try:
                    value = items[name]
                except KeyError:
                    raise ValueError(f"Item '{name}' is missing for log")
                if isinstance(value, float):
                    value = '{:.3f}'.format(value)
                log_items.append(f"{name}: {value}")
            log_line = ' | '.join(log_items)
            print(log_line)

    def pprint(self, metrics:dict, line=True):
        msg = []
        for k, v in metrics.items():
            if isinstance(v, float):
                v = '{:.2f}'.format(v)
            msg.append(f"{k}={v}")
        if line:
            return " | ".join(msg)
        return "\n".join(msg)

    def print_conf_matrix(self, conf_matrix:Tensor):
        n_classes = conf_matrix.shape[0]
        if n_classes > 2:
            print("...print conf matrix...")
        else:
            print("...print bin conf matrix...")


if __name__ == '__main__':
    metrics_computer = BinClassificationMetrics(
        compute_metrics=["Acc", "Precision", "Recall"],
    )
    probs = torch.tensor(
        [[.1, .4, .7, .9, .2, 0, .2],
         [.9, .6, .3, .1, .8, 1, .8]]
    ).transpose(0, 1)
    targ = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 0],
         [1, 0, 0, 0, 0, 0, 1]]
    ).transpose(0, 1)
    metrics = metrics_computer.compute(probs, targ)
    print(metrics)
