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
            dict: {'{NameLoss}': torch.Tensor(1) mean loss across batch}
        NOTE: '{NameLoss}' must end with '*Loss'
        """
        pass


class CrossEntropyLoss(Loss):
    def __init__(self, device='cpu', weights=None):
        if isinstance(weights, list):
            weights = torch.tensor(weights)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        self.loss = self.loss.to(device)

    def __call__(self, pred: Tensor, targ: Tensor) -> dict:
        loss = self.loss(pred, targ)
        name = "CrossEntropyLoss"
        return {name: loss}


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
        n_classes (int): number of classes C
        compute_metrics (list[str]): order of metrics to compute
        """
        self.compute_metrics = compute_metrics
        self.metrics_funcs = {
            "TP": self.tp,
            "FN": self.fn,
            "TN": self.tn,
            "FP": self.fp,

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
    def tp(pred:torch.Tensor, targ:torch.Tensor) -> float:
        """
        True positive
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: true positive
        """
        tp = torch.sum(torch.logical_and(pred == 1, targ == 1)).item()
        return tp
    
    @staticmethod
    def tn(pred:torch.Tensor, targ:torch.Tensor) -> float:
        """
        True negative
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: true negative
        """
        tn = torch.sum(torch.logical_and(pred == 0, targ == 0)).item()
        return tn
    
    @staticmethod
    def fp(pred:torch.Tensor, targ:torch.Tensor) -> float:
        """
        False positive
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: false positive
        """
        fp = torch.sum(torch.logical_and(pred == 1, targ == 0)).item()
        return fp
    
    @staticmethod
    def fn(pred:torch.Tensor, targ:torch.Tensor) -> float:
        """
        False positive
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
        Returns:
            int: false negative
        """
        fn = torch.sum(torch.logical_and(pred == 0, targ == 1)).item()
        return fn

    def compute(self, probs:torch.Tensor, targ:torch.Tensor,
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

        metrics = OrderedDict()
        for name in self.compute_metrics:
            try:
                metrics[name] = self.metrics_funcs[name](pred, targ)
            except KeyError:
                continue
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
