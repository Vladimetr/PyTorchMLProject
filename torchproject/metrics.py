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


def init_loss(loss_cfg:dict, device='cpu') -> Loss:
    loss_cfg = dict(loss_cfg)  # copy
    loss_class = loss_cfg.pop("class")
    if loss_class == 'cross_entropy':
        loss = CrossEntropyLoss(**loss_cfg, device=device)
        
    # another loss
    else:
        raise ValueError(f"Invalid loss '{loss_class}'")
    return loss


# all valid metrics
METRICS = ["TP", "FN", "FP", "TN", 
           "acc", "recall", "precision", 
           "conf_matrix",
]
# metrics that are defined for particular class
CLASS_METRICS = ["TP", "FN", "FP", "TN",
                 "recall", "precision"
                 ]
CSV_SEP = ' '


class ClassificationMetrics:
    def __init__(self, classes:List[str], metrics:List[str],
                 logger=None, epoch=False, step=False,
                 log_title=True):
        """
        classes (list[str]): C-list of class names
        metrics (list[str]): order of metrics to compute
        epoch (bool): whether to log epoch
        step (bool): whether to log step
        log_title (bool): whether log column names
        """
        self.compute_metrics = metrics 

        # Check given metrics are in valid list
        for metric_name in metrics:
            if not metric_name in METRICS:
                raise ValueError(f"Invalid metric '{metric_name}'")

        # All of these metrics are computed based on conf matrix
        self.metrics_funcs = {
            "TP": self.tp,
            "FN": self.fn,
            "TN": self.tn,
            "FP": self.fp,
            "acc": self.accuracy,
            "recall": self.recall,  # TPR
            "precision": self.precison,
        }

        # define metrics logging format
        self.classes = classes
        self.n_classes = len(classes)
        self.epoch = epoch
        self.step = step
        self.logger = logger
        self._init_log_format(log_title=log_title)
        self.reset_summary()

    def _init_log_format(self, log_title=True):
        log_items = []
        for metric_name in self.compute_metrics:
            if metric_name in CLASS_METRICS:
                log_items += [f"{metric_name}-{class_name}" 
                              for class_name in self.classes]
            else:
                log_items.append(metric_name)
                
        if self.step:
            log_items = ['step'] + log_items
        if self.epoch:
            log_items = ['epoch'] + log_items
        self.log_items = log_items
        if self.logger and log_title:
            # log title
            self.logger.info(CSV_SEP.join(log_items))

    def conf_matrix(self, pred:Tensor, targ:Tensor) -> Tensor:
        """
        C - n_classes
             0    2    3
        0 |    |    |    |
        2 |    |    |    |
        3 |    |    |    |
        columns: actual
        rows: predicted
        Args:
            pred (B, ): indexes of pred classes
            targ (B, ): indexes of targ classes
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

    def tp(self, class_indx:int,
           pred:Tensor, targ:Tensor, 
           conf_matrix:Tensor=None) -> int:
        """
        True positive
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            int: false positive
        """
        if conf_matrix is None:
            conf_matrix = self.conf_matrix(pred, targ)  # (C, C)
        tp = conf_matrix[class_indx, class_indx].item()
        return tp
    
    @staticmethod
    def tn(class_indx:int,
           pred:Tensor, targ:Tensor, 
           conf_matrix:Tensor=None) -> int:
        """
        True negative
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            int: true negative
        """
        raise NotImplementedError()
    
    def fp(self, class_indx:int,
           pred:Tensor, targ:Tensor, 
           conf_matrix:Tensor=None) -> int:
        """
        False positive
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            int: false positive
        """
        if conf_matrix is None:
            conf_matrix = self.conf_matrix(pred, targ)  # (C, C)
        
        tp = conf_matrix[class_indx, class_indx].item()
        pred_sum = conf_matrix[class_indx, :].sum().item()
        fp = pred_sum - tp
        return fp
    
    def fn(self, class_indx:int,
           pred:Tensor, targ:Tensor, 
           conf_matrix:Tensor=None) -> int:
        """
        False negative
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            int: false negative
        """
        if conf_matrix is None:
            conf_matrix = self.conf_matrix(pred, targ)  # (C, C)
        
        tp = conf_matrix[class_indx, class_indx].item()
        targ_sum = conf_matrix[:, class_indx].sum().item()
        fn = targ_sum - tp
        return fn
    
    def accuracy(self, 
                 pred:Tensor, targ:Tensor, 
                 class_indx:int=None,
                 conf_matrix:Tensor=None) -> float:
        """ TP / Total 
        Args:
            class_index (int): for which class compute this metric
                If None, compute total accuracy
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            float: false negative
        """
        if conf_matrix is None:
            conf_matrix = self.conf_matrix(pred, targ)  # (C, C)

        if class_indx is not None:
            tp = conf_matrix[class_indx, class_indx]
            targ_sum = conf_matrix[:, class_indx].sum().item()
            pred_sum = conf_matrix[class_indx, :].sum().item()
            acc = targ_sum + pred_sum - 2 * tp
            return acc
        
        # total accuracy
        tp = torch.diagonal(conf_matrix).sum().item()
        total = conf_matrix.sum().item()
        acc = tp / total
        return acc
    
    def precison(self, class_indx:int,
                 pred:Tensor, targ:Tensor, 
                 conf_matrix:Tensor=None) -> float:
        """ TPR = TP / (TP + FP) 
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            float: precision
        """
        tp = self.tp(class_indx, pred, targ, conf_matrix)
        fp = self.fp(class_indx, pred, targ, conf_matrix)
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            prec = -1
        return prec
    
    def recall(self, class_indx:int,
               pred:Tensor, targ:Tensor, 
               conf_matrix:Tensor=None) -> float:
        """ TPR = TP / (TP + FN) 
        Args:
            class_index (int): for which class compute this metric
            pred (B, ): pred classes 
            targ (B, ): targ classes
        NOTE: for conf matrix: rows - preds, colums - targs
        Returns:
            int: recall
        """
        tp = self.tp(class_indx, pred, targ, conf_matrix)
        fn = self.fn(class_indx, pred, targ, conf_matrix)
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = -1
        return recall
    
    def from_conf_matrix(self, metrics:List[str], conf_matrix:Tensor
                         ) -> dict:
        """ Extract given metrics from confusion matrix
        """
        result = OrderedDict()
        for metric_name in metrics:
            if metric_name in CLASS_METRICS:
                for class_i, class_name in enumerate(self.classes):
                    func = self.metrics_funcs[metric_name]
                    name = f"{metric_name}-{class_name}"
                    result[name] = func(pred=None, targ=None,
                                        class_indx=class_i,
                                        conf_matrix=conf_matrix)
            else:
                func = self.metrics_funcs[metric_name]
                result[metric_name] = func(pred=None, targ=None,
                                           conf_matrix=conf_matrix)
        return result


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
        # class indexes with max prob (don't use threshold here)
        pred = torch.max(probs, dim=1)[1]  # (B, )
        
        # Confusion matrix is core of following metrics
        metrics = list(self.compute_metrics)  # copy
        conf_matrix = self.conf_matrix(pred=pred, targ=targ)
        if "conf_matrix" in metrics:
            result["conf_matrix"] = conf_matrix
            metrics.remove("conf_matrix")

        result = self.from_conf_matrix(metrics, conf_matrix)

        if accumulate:
            # accumulating confusion matrix is enough
            self.sum_conf_matrix += conf_matrix
        return result
    
    def summary(self, metrics:List[str]) -> dict:
        """ Get summary of accumulated metrics 
        """
        result = OrderedDict()
        metrics = list(metrics)  # copy
        if "conf_matrix" in metrics:
            result["conf_matrix"] = self.sum_conf_matrix
            metrics.remove("conf_matrix")

        result.update(self.from_conf_matrix(metrics,
                                            self.sum_conf_matrix))
        return result
    
    def reset_summary(self):
        self.sum_conf_matrix = torch.zeros(self.n_classes, 
                                           self.n_classes,
                                           dtype=torch.int)

    def log_metrics(self, metrics:dict, epoch:int=None, step:int=None):
        items = dict(metrics)
        if epoch and 'epoch' in self.log_items:
            items['epoch'] = epoch
        if step and 'step' in self.log_items:
            items['step'] = step

        if self.logger:
            # log to file in .csv format
            log_line = CSV_SEP.join(str(items[col]) for col in self.log_items)
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

    def pprint(self, metrics:dict, line=True, 
               with_conf_matrix:bool=False):
        msg = []
        metrics = dict(metrics)  # copy
        conf_matrix = metrics.pop("conf_matrix", None)
        for k, v in metrics.items():
            if isinstance(v, float):
                v = '{:.2f}'.format(v)
            msg.append(f"{k}={v}")
        if line:
            print(" | ".join(msg))
        print("\n".join(msg))
        if with_conf_matrix and conf_matrix is not None:
            self.pprint_conf_matrix(conf_matrix)

    def pprint_conf_matrix(self, conf_matrix:Tensor):
        max_class_name = max([len(cls_name) for cls_name in self.classes])
        cell_size = max_class_name + 2  #  "-dog-"
        row_title = ' ' * cell_size + '|'
        for class_name in self.classes:
            row_title += ("{:^" + str(cell_size) + "}|")\
                         .format(class_name)
            
        out_str = row_title
        for j in range(self.n_classes):
            row = ("{:^" + str(cell_size) + "}|")\
                  .format(self.classes[j])
            for i in range(self.n_classes):
                row += ("{:^" + str(cell_size) + "}|")\
                         .format(conf_matrix[j, i])
            out_str += "\n" + row

        print(out_str)



if __name__ == '__main__':
    metrics_computer = ClassificationMetrics(
        metrics=["acc", "precision", "recall"],
        classes=["cat", "dog", "bear"],
        epoch=True, step=True,
    )
    probs = torch.tensor(
        [[.1, .4, .7, .5, .2, 0, .2],
         [.8, .3, .1, .1, .2, 1, .3],
         [.1, .3, .2, .4, .6, 0, .5]]
    ).transpose(0, 1)
    targ = torch.tensor([0, 2, 1, 2, 1, 1, 0])

    metrics = metrics_computer.compute(probs, targ, accumulate=True)
    metrics_computer.log_metrics(metrics, epoch=1, step=1)
    metrics_computer.log_metrics(metrics, epoch=1, step=2)
    metrics_computer.log_metrics(metrics, epoch=1, step=3)

    sum_metrics = metrics_computer.summary(["acc", "precision", "recall", "conf_matrix"])
    print(metrics_computer.pretty(sum_metrics))
    metrics_computer.pprint_conf_matrix(sum_metrics["conf_matrix"])
