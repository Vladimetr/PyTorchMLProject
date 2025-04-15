from collections import OrderedDict
from typing import List, Dict
import sys
from abc import ABCMeta, abstractmethod
import torch
from  torch import Tensor
from .utils.manager import ClearMLManager


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
    def __call__(self, pred:Tensor, targ:Tensor,
                 rn_indices:Tensor=None, lam:Tensor=None) -> dict:
        """
        B - batch size
        C - n classes
        NOTE: some losses support mixup method
        https://github.com/fschmid56/EfficientAT/blob/a425fdce92572e602a1d5634799bd9f1f2efa806/ex_esc50.py#L103
        'rn_indices' and 'lam' are provided for this
        Args:
            pred (B, C): predicted logits
            targ (B, C)): target one hot
            rn_indices (): random indices
                None by default
            lam (): lambda. None by default
            If your loss doesn't support mixup
            raise NotImplementedError()
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
        self.loss = torch.nn.CrossEntropyLoss(weight=weights,
                                              reduction="none")
        self.loss = self.loss.to(device)

    def __call__(self, pred: Tensor, targ: Tensor,
                 rn_indices:Tensor=None, lam:Tensor=None) -> tuple:
        if rn_indices is not None:
            # mixup
            bs = pred.shape[0]
            loss1 = self.loss(pred, targ)  # (B, C)
            loss2 = self.loss(pred, targ[rn_indices])  # (B, C)
            loss = loss1 * lam.reshape(bs) + loss2 * (1. - lam.reshape(bs))
            loss = loss.mean()
            ce_value = loss.item()
            loss_values = {
                "CrossEntropyLoss": ce_value,
            }
            return loss, loss_values

        # no mixup
        loss = self.loss(pred, targ).mean()
        ce_value = loss.item()  # float
        loss_values = {
            "CrossEntropyLoss": ce_value,
        }
        return loss, loss_values


class MixupCrossEntropyLoss(Loss):
    def __init__(self, device='cpu', weights=None):
        if isinstance(weights, list):
            weights = Tensor(weights)
        self.loss = torch.nn.CrossEntropyLoss(weight=weights, 
                                              reduction="none")
        self.loss = self.loss.to(device)

    def __call__(self, pred: Tensor, targ: Tensor,
                 rn_indices:Tensor, lam:Tensor) -> tuple:
        bs = pred.shape[0]
        loss1 = self.loss(pred, targ)  # (B, C)
        loss2 = self.loss(pred, targ[rn_indices])  # (B, C)
        loss = loss1 * lam.reshape(bs) + loss2 * (1. - lam.reshape(bs))
        loss = loss.mean()
        ce_value = loss.item()
        loss_values = {
            "CrossEntropyLoss": ce_value,
        }
        return loss, loss_values


def init_loss(loss_cfg:dict,
              device:str="cpu"
              ) -> Loss:
    """
    loss_cfg (dict): 
        {
            "{class_name}": kwargs (dict)
        }
    """
    name = loss_cfg["use"]
    params = loss_cfg[name]
    try:
        # define class
        loss = globals()[name]
    except KeyError:
        raise ValueError(f"Invalid loss name '{name}'")
    try:
        # init
        loss = loss(**params, device=device)
    except TypeError:
        raise ValueError(f"Invalid loss params {params}")
    return loss


# metrics that can be computed on each step 
# they are based on confusion matrix
CM_METRICS = ["conf_matrix",
              "TP", "FN", "FP", "TN", 
              "acc", "recall", "precision" 
               ]
# metrics that are defined for particular class
CLASS_METRICS = ["TP", "FN", "FP", "TN",
                 "recall", "precision"
                 ]
# plots based on probs (not preds) and targets
# not able for step metrics
PLOTS = ["PR-curve", "ROC-curve"]
CSV_SEP = ' '


class ClassificationMetrics:
    def __init__(self, classes:List[str], 
                 step_metrics:List[str]=[],
                 summary_metrics:List[str]=[],
                 logger=None, epoch=False, step=False,
                 log_title=True):
        """
        classes (list[str]): C-list of class names
        metrics (list[str]): order of metrics to compute
                             in method `compute()`
        epoch (bool): whether to log epoch
        step (bool): whether to log step
        log_title (bool): whether log column names
            Log title can be skipped if this is a resume
            of existed experiment
        """
        assert not any([m in PLOTS for m in step_metrics]), \
            f"Step metrics can not be from {PLOTS}"
        # Get step metrics list that will be computed
        # based on conf matrix (CM)
        self.cm_metrics = list(filter(lambda x: x in CM_METRICS, 
                                      step_metrics))
        self.summary_metrics = summary_metrics

        # All of these metrics are computed based on conf matrix (CM)
        self.cm_metrics_funcs = {
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
        self._init_log_format(step_metrics, log_title=log_title)
        self._init_summary(summary_metrics)

    def _init_summary(self, metrics:List[str]):
        self.accumulate_matrix = False
        self.accumulate_probs = False  # for PLOTS
        self.avg_metrics : Dict[str, list] = dict()
        for metric in metrics:
            # confusion matrix metrics
            if metric in CM_METRICS:
                self.sum_conf_matrix = torch.zeros(self.n_classes, 
                                                   self.n_classes,
                                                   dtype=torch.int)
                self.accumulate_matrix = True
            elif metric in PLOTS:
                self.sum_probs, self.sum_targs = [], []
                self.accumulate_probs = True
            else:
                self.avg_metrics[metric] = []
                # avg metric = sum of list items / N
        self.summary_metrics = metrics

    def _init_log_format(self, metrics:List[str], log_title=True):
        """ Create log format for writing step metrics
        and define title in it
        epoch | step | metric1-name | metric2-name |
          1   |   1  |    0.479     | 0.979
        NOTE: if logger file is defined
        these files can be parsed for plotting
        step metrics in TensorBoard as example
        """
        log_items = []
        for metric_name in metrics:
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
    
    def tn(self, class_indx:int,
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
        if conf_matrix is None:
            conf_matrix = self.conf_matrix(pred, targ)  # (C, C)
        s = conf_matrix.sum().item()
        tp = conf_matrix[class_indx, class_indx].item()
        pred_sum = conf_matrix[class_indx, :].sum().item()
        targ_sum = conf_matrix[:, class_indx].sum().item()
        tn = s - pred_sum - targ_sum + tp
        return tn
    
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
        Args:
            metrics (list[str]): list of metric names within CM_METRICS
            conf_matrix (Tensor): (C, C) confusion matrix
        Returns:
            dict: keys are metrics from given list
        """
        result = OrderedDict()
        for metric_name in metrics:
            if metric_name == "conf_matrix":
                result["conf_matrix"] = conf_matrix
            elif metric_name in CLASS_METRICS:
                # compute this metric for each class
                for class_i, class_name in enumerate(self.classes):
                    func = self.cm_metrics_funcs[metric_name]
                    name = f"{metric_name}-{class_name}"
                    result[name] = func(pred=None, targ=None,
                                        class_indx=class_i,
                                        conf_matrix=conf_matrix)
            else:
                func = self.cm_metrics_funcs[metric_name]
                result[metric_name] = func(pred=None, targ=None,
                                           conf_matrix=conf_matrix)
        return result

    def step_metrics(self, probs:Tensor, targ:Tensor,
                     add_summary=False, 
                     precomputed:dict={}) -> dict:
        """
        Compute defined metrics after new train/test step
        B - batch size
        C - n classes
        Args:
            probs (B, C): probs for each class
            targ (B, )): target indexes class
            add_summary (bool): whether to add summary
            precomputed: dict with precomputed metrics
                (for ex. losses) that can be logged or summarized
        Returns:
            dict: dict with `step_metrics`
        """
        if probs.shape[0] != targ.shape[0]:
            raise ValueError("Mismatch probs and targ shapes")
        if probs.shape[1] != self.n_classes:
            raise ValueError(f"Invalid number of classes {probs.shape[1]}")
        # class indexes with max prob (don't use threshold here)
        pred = torch.max(probs, dim=1)[1]  # (B, )
        
        # Compute metrics that are based on confusion matrix
        conf_matrix = None
        if self.cm_metrics:
            conf_matrix = self.conf_matrix(pred, targ)
            metrics = self.from_conf_matrix(self.cm_metrics, conf_matrix)
        else:
            metrics = OrderedDict()

        # add precomputed metrics
        metrics.update(precomputed)

        if add_summary:
            self.add_summary(probs, targ, conf_matrix, precomputed)

        return metrics
    
    def add_summary(self, 
                    probs:Tensor=None, targ:Tensor=None,
                    conf_matrix:Tensor=None,
                    precomputed:dict={}):
        # for confusion matrix metrics
        if self.accumulate_matrix:
            if conf_matrix is None:
                pred = torch.max(probs, dim=1)[1]
                conf_matrix = self.conf_matrix(pred, targ)
            self.sum_conf_matrix += conf_matrix
        # for plots
        if self.accumulate_probs:
            self.sum_probs.append(probs)
            self.sum_targs.append(targ)
        # precomputed
        for metric, values in self.avg_metrics.items():
            try:
                value = precomputed[metric]
            except KeyError:
                raise Exception(f"Metric {metric} must be precomputed "\
                                "for summary")
            values.append(value)

    def get_summary(self) -> dict:
        """ Get summary of accumulated metrics 
        NOTE: plots are built using other methods
        Returns:
            dict: dict with `summary_metrics`
        """
        metrics = OrderedDict()

        # confusion matrix summary metrics
        cm_sum_metrics = list(filter(lambda x: x in CM_METRICS, 
                                     self.summary_metrics))
        
        if cm_sum_metrics:
            a = self.from_conf_matrix(cm_sum_metrics, self.sum_conf_matrix)
            metrics.update(a)

        # avg precomputed metrics
        for metric, values in self.avg_metrics.items():
            avg = sum(values) / len(values)
            metrics[metric] = avg

        return metrics
    
    def reset_summary(self):
        self.sum_conf_matrix = torch.zeros(self.n_classes, 
                                           self.n_classes,
                                           dtype=torch.int)
        self.sum_probs = []
        # for each sample probs [(B, C)]
        self.sum_targs = []
        # for each sample targets [(B, )]
        for metric in self.avg_metrics.keys():
            self.avg_metrics[metric] = []

    def log_metrics(self, metrics:dict, epoch:int=None, step:int=None):
        """ Write given metrics to stdout 
        or file if logger was defined
        """
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
               with_conf_matrix:bool=False,
               duplicate_file:str=None):
        """
        duplicate_file (str): file to duplicate pprint
        """
        msg = []
        metrics = dict(metrics)  # copy
        conf_matrix = metrics.pop("conf_matrix", None)
        for k, v in metrics.items():
            if isinstance(v, float):
                v = '{:.2f}'.format(v)
            msg.append(f"{k}={v}")
        output = " | ".join(msg) if line else "\n".join(msg)
        print(output)
        if duplicate_file:
            with open(duplicate_file, 'a') as f:
                print(output, file=f)
        if with_conf_matrix and conf_matrix is not None:
            self.pprint_conf_matrix(conf_matrix, duplicate_file=duplicate_file)

    def pprint_conf_matrix(self, conf_matrix:Tensor, 
                           duplicate_file:str=None):
        """
        duplicate_file (str): file to duplicate pprint
        """
        max_class_name = max([len(cls_name) \
                              for cls_name in self.classes])
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

        n_classes = conf_matrix.shape[0]
        if n_classes > 10:
            print("Number of classes is too large for print confusion matrix")
        else:
            print(out_str)
        if duplicate_file:
            with open(duplicate_file, 'a') as f:
                print(out_str, file=f)

    def plot_pr(self, class_:str, manager:ClearMLManager):
        """
        Plot Precision-Recall curve in ClearML
        based on summary (accumulated) probs and targets
        """
        try:
            cls_indx = self.classes.index(class_)
        except ValueError:
            raise ValueError(f"Unknown class name '{class_}'")
        sum_probs = torch.concat(self.sum_probs, dim=0)
        # (M, C)
        cls_probs = sum_probs[:, cls_indx]
        
        sum_targs = torch.concat(self.sum_targs, dim=0)
        # (M, )
        cls_targs = (sum_targs == cls_indx).to(torch.int8)

        manager.plot_pr_curve(cls_probs, cls_targs, 
                              pos_label=None,
                              class_name=class_)
        
    def plot_roc(self, class_:str, manager:ClearMLManager):
        """
        Plot ROC curve in ClearML
        based on summary (accumulated) probs and targets
        """
        try:
            cls_indx = self.classes.index(class_)
        except ValueError:
            raise ValueError(f"Unknown class name '{class_}'")
        sum_probs = torch.concat(self.sum_probs, dim=0)
        # (M, C)
        cls_probs = sum_probs[:, cls_indx]
        
        sum_targs = torch.concat(self.sum_targs, dim=0)
        # (M, )
        cls_targs = (sum_targs == cls_indx).to(torch.int8)

        manager.plot_roc_curve(cls_probs, cls_targs, 
                               pos_label=None,
                               class_name=class_)




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
