from collections import OrderedDict
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
            pred (B, C):
            targ (B, )): 
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
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
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
                 epoch=False, step=False):
        """
        compute_metrics (list[str]): order of metrics to compute
        epoch (bool): whether to log epoch
        step (bool): whether to log step
        """
        self.compute_metrics = compute_metrics
        self.metrics_funcs = {
            "TP": self.tp,
            "FN": self.fn,
            "TN": self.tn,
            "FP": self.fp,
            "Acc": self.accuracy,
            "Recall": self.recall,  # TPR
            "Precision": self.precison,

        }
        self.reset_summary()
        log_items = list(compute_metrics)
        if step:
            log_items = ['step'] + log_items
        if epoch:
            log_items = ['epoch'] + log_items
        self.log_items = log_items
        if logger:
            # set title
            logger.info(' '.join(log_items))
        self.logger = logger
        self.epoch = epoch
        self.step = step
        
    @staticmethod
    def tp(pred:Tensor, targ:Tensor,
           precomputed:dict=None) -> int:
        """
        True positive
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: true positive
        """
        try:
            conf_matrix = precomputed["conf_matrix"]
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
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: true negative
        """
        try:
            conf_matrix = precomputed["conf_matrix"]
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
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: false positive
        """
        try:
            conf_matrix = precomputed["conf_matrix"]
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
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: false negative
        """
        try:
            conf_matrix = precomputed["conf_matrix"]
        except (TypeError, KeyError):
            fn = torch.sum(torch.logical_and(pred == 0, targ == 1)).item()
        else:
            fn = conf_matrix[0, 1].item()
        return fn

    def conf_matrix(self, pred:Tensor, targ:Tensor,
                    precomputed:dict=None) -> Tensor:
        """
        | TN | FN |
        | FP | TP |
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
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

    def accuracy(self, pred:Tensor, targ:Tensor, 
                 precomputed:dict=None) -> float:
        """ (TP + TN) / Total """
        try:
            conf_matrix = precomputed["conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.conf_matrix(pred, targ)
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
            conf_matrix = precomputed["conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.conf_matrix(pred, targ)
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
            conf_matrix = precomputed["conf_matrix"]
        except (TypeError, KeyError):
            conf_matrix = self.conf_matrix(pred, targ)
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
        C - n classes = 2
        Args:
            probs (B, C): probs for each class
            targ (B, C)): one-hot of target class
        Returns:
            dict: dict with metrics
        """
        # class indexes with max prob
        pred = torch.max(probs, dim=1)[1]  # (B, )
        # one-hot to indexes
        targ = torch.argmax(targ, dim=1)  # (B, )
        conf_matrix = self.conf_matrix(pred, targ)
        metrics = OrderedDict({
            "conf_matrix": conf_matrix
        })
        for name in self.compute_metrics:
            try:
                func = self.metrics_funcs[name]
                metrics[name] = func(pred=None, targ=None,
                                     precomputed=metrics)
            except KeyError:
                continue
        metrics.pop("conf_matrix")
        if accumulate:
            self.add_summary(metrics)
        return metrics
    
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

    def add_summary(self, metrics:dict):
        for k, v in metrics.items():
            try:
                self.sum_metrics[k].append(v)
            except KeyError:
                self.sum_metrics[k] = [v]

    def summary(self) -> dict:
        """ Get average of accumulated metrics """
        avg_metrics = dict()
        for k, v in self.sum_metrics.items():
            avg_metrics[k] = sum(v) / len(v)
        return avg_metrics
    
    def reset_summary(self):
        self.sum_metrics = dict()

    def pprint(self, metrics:dict, line=True):
        msg = []
        for k, v in metrics.items():
            if isinstance(v, float):
                v = '{:.2f}'.format(v)
            msg.append(f"{k}={v}")
        if line:
            return " | ".join(msg)
        return "\n".join(msg)


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
