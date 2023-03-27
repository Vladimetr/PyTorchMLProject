from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import torch


class Loss(metaclass=ABCMeta):
    """ Abstract Loss """
    @abstractmethod
    def __init__(self, n_classes, device='cpu'):
        """
        Base parameters
        n_classes (int): number of classes
        device (str): 'cpu' or 'cuda'
        """
        pass

    @abstractmethod
    def __call__(self, pred:torch.Tensor, targ:torch.Tensor):
        """
        B - batch size
        C - n classes
        Args:
            pred (B, C):
            targ (B, )): 
        Returns:
            torch.Tensor (1): mean loss across batch
        """
        pass


def init_loss(name:str, params:dict, device='cpu') -> Loss:
    if name == 'cross_entropy':
        params["pos_weight"] = torch.tensor(params["pos_weight"])
        loss = torch.nn.BCEWithLogitsLoss(**params).to(device)
        
    # another loss
    else:
        raise ValueError(f"Invalid loss '{name}'")
    return loss



class Metrics:
    def __init__(self, n_classes:int, compute_metrics:list):
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

    @staticmethod
    def tp(pred:torch.Tensor, target:torch.Tensor) -> float:
        """
        True positive
        """
        return 0.7
    
    @staticmethod
    def tn(pred:torch.Tensor, target:torch.Tensor) -> float:
        """
        True negative
        """
        return 0.6
    
    @staticmethod
    def fp(pred:torch.Tensor, target:torch.Tensor) -> float:
        """
        False positive
        """
        return 0.2
    
    @staticmethod
    def fn(pred:torch.Tensor, target:torch.Tensor) -> float:
        """
        False negative
        """
        return 0.1

    def compute(self, probs:torch.Tensor, targ:torch.Tensor,
                accumulate=False) -> dict:
        """
        B - batch size
        C - n classes
        Args:
            probs (B, C): probs for each class
            targ (B, )): index of target class
        Returns:
            dict: dict with metrics
        """
        pred = torch.max(probs, dim=1)  # (B, )
        metrics = OrderedDict()
        for name in self.compute_metrics:
            try:
                metrics[name] = self.metrics_funcs[name](pred, targ)
            except KeyError:
                raise ValueError(f"Invalid metric name '{name}'")
        if accumulate:
            self.add_summary(metrics)
        return metrics
    
    def add_summary(self, metrics:dict):
        for k, v in metrics:
            try:
                self.sum_metrics[k].append(v)
            except KeyError:
                self.sum_metrics[k] = [v]

    def summary(self) -> dict:
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
