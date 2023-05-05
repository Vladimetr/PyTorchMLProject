import os.path as osp
import mlflow
from clearml import Task, Logger, TaskTypes
from typing import Union, List
import torch
import numpy as np

PROJECT_NAME = 'VAD'

matrix = Union[np.ndarray, torch.Tensor]

class BaseManager:
    def __init__(self, train=True):
        """
        train (bool): mode of experiment (train or test)
        """
        pass

    def add_tags(self, tags:Union[List[str], dict], rewrite=False):
        """
        Tag might be str, like ['best']
        or dict, like {'mode': 'train'}
        Args:
            rewrite (bool): whether to delete previous tags
        """
        pass

    def log_step_metrics(self, metrics:dict, step:int):
        """
        Log epoch metrics or test step metrics for plot
        NOTE: Don't log a lot of metrics (more than 200 per experiment),
        because it can slow down UI
        NOTE: Loss can be stored in metrics dict with key "*Loss"
        (for ex. "CrossEntropyLoss")
        """
        pass

    def log_summary_metrics(self, metrics:dict):
        """
        Log average or best metrics
        """
        pass

    def log_hyperparams(self, hparams:dict) -> dict:
        """ Hyperparams are defined in config manager:hparams
        NOTE: Some managers like ClearML allow to update hparams
        in UI. So this method returns updated hparams to set them
        in train.py
        """
        pass

    def log_config(self, config:Union[dict, str], name='config.yaml'
                   ) -> Union[dict, str]:
        """
        Log config dict or config.yaml
        NOTE: Some managers like ClearML allow to update config
        in UI. So this method returns updated config to set it
        in train.py
        """
        pass

    def log_metadata(self, metadata:dict):
        """ Additional static (unchangable) metadata like
        - experiment storage
        - train steps
        """
        pass

    def set_status(self, status:str):
        """
        Some managers support statuses, like
        "Finished" or "Failed"
        Args:
            status (str)
        """
        pass

    def set_iterations(self, iterations:int):
        """
        Set number of train steps (for ex, epochs)
        or test steps (number of test batches)
        """
        pass

    def log_iteration(self, iteration:int):
        """
        Log iteration of train (for ex, epoch)
        or test step (batch index)
        """
        pass

    def close(self):
        """
        Some managers require closing
        """
        pass

    def log_confusion_matrix(self, conf_matrix: matrix, 
                             classes:List[str]=None, step:int=None):
        """
        Some managers support logging confusion matrix
        NOTE: xaxis="target" and yaxis="predict"
        """
        # validate confusion matrix
        shape = conf_matrix.shape
        if classes:
            n_classes = len(classes)
        else:
            n_classes = shape[0]
            classes = [str(i) for i in range(n_classes)]
        if not (len(shape) == 2 and shape[0] == shape[1] == n_classes):
            raise ValueError('Mismatch matrix shape and N classes')

    def log_roc_curve(self, data):
        """
        Some managers support logging ROC curve
        """
        pass
        

class ClearMLManager(BaseManager):
    def __init__(self, key_token:str, secret_token:str, 
                 subproject:bool=False,  # experiment format
                 experiment='noname', 
                 run_name:str="noname", 
                 train:bool=True, 
                 tags:dict={}):
        Task.set_credentials(key=key_token, secret=secret_token)
        task_type = TaskTypes.training if train else TaskTypes.testing
        if subproject:
            project_name = PROJECT_NAME + '/' + experiment
            task_name = run_name
        else:
            project_name = PROJECT_NAME
            task_name = experiment + '_' + run_name
        self.task = Task.init(project_name=project_name, 
                              task_name=task_name, 
                              task_type=task_type,
                              auto_connect_frameworks=False)
        self.task.rename(task_name)  # if it was renamed
        # Turn off auto saveing ML models and other artifacts
        self.logger = Logger.current_logger()
        self.max_step = 0
        self.add_tags(tags)
        print(f"ClearML experiment: '{project_name}/{task_name}'")
        
    def log_hyperparams(self, hparams: dict) -> dict:
        hparams = self.task.connect(hparams, name='hparams')
        return hparams

    def log_config(self, config: Union[dict, str]) -> Union[dict, str]:
        config = self.task.connect_configuration(config, 
                                                 name='config.yaml')
        return config
    
    def log_metadata(self, metadata:dict):
        self.task.set_configuration_object(name='meta.yaml', 
                                           config_dict=metadata)

    def log_step_metrics(self, metrics: dict, step: int):
        for metric_name, value in metrics.items():
            title = 'Metrics'
            if 'Loss' in metric_name:
                title = 'Loss'
                metric_name = metric_name.replace('Loss', '')
            self.logger.report_scalar(
                title=title, 
                series=metric_name, 
                value=value, 
                iteration=step
        )
        self.max_step = max(self.max_step, step)
            
    def log_summary_metrics(self, metrics: dict):
        for metric_name, value in metrics.items():
            self.logger.report_single_value(metric_name, value)
            
    def add_tags(self, tags:Union[List[str], dict], rewrite=False):
        if isinstance(tags, dict):
            tags = tags.values()
        tags = list(map(str, tags))
        if rewrite:
            self.task.set_tags(tags)
        else:
            self.task.add_tags(tags)

    def close(self):
        self.task.close()

    def log_confusion_matrix(self, conf_matrix: matrix, 
                             classes:List[str]=None, 
                             title:str='Confusion matrix',
                             step:int=None):
        # validate input
        super().log_confusion_matrix(conf_matrix, classes)
        step = step or self.max_step
        if isinstance(conf_matrix, torch.Tensor):
            conf_matrix = conf_matrix.numpy()
        self.logger.report_confusion_matrix(
                    title, "ignored", 
                    iteration=step, matrix=conf_matrix,
                    xlabels=classes, ylabels=classes,
                    xaxis="target", yaxis="predict"
        )


class MLFlowManager(BaseManager):
    def __init__(self, url:str, 
                 experiment='noname', 
                 run_name:str="noname",
                 train:bool=True, 
                 tags:dict={}):
        """
        url (str): For ex. 'http://192.168.11.181:3500'
        id (int): number of experiment run (Train-001)
        NOTE: MLflow experiments storage is '/mlflow/mlruns/'
        """
        mlflow.set_tracking_uri(url)
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            raise ValueError(f"Experiment '{experiment}' doesn't exists. "\
                             f"Please, create it {url}")
        self.exp_id = exp.experiment_id
        assert isinstance(self.exp_id, str)

        run_id = None  # to create new one
        tags['mode'] = 'train' if train else 'test'
        run = mlflow.start_run(
            experiment_id=self.exp_id, 
            run_id=run_id, 
            run_name=run_name, 
            tags=tags, 
            description="my example"
        )
        self.run_id = run.info.run_id
        self.max_step = 0
        self.train = train
        print(f"MLFlow experiment: '{experiment}/{run_name}'")

    def log_step_metrics(self, metrics:dict, step:int):
        mlflow.log_metrics(metrics, step=step)
        self.max_step = max(self.max_step, step)

    def log_hyperparams(self, hparams:dict) -> dict:
        mlflow.log_params(hparams)
        return hparams

    def log_summary_metrics(self, metrics:dict):
        mlflow.log_metrics(metrics, step=self.max_step + 1)
    
    def log_metadata(self, metadata: dict):
        mlflow.log_dict(metadata, 'meta.yaml')

    def log_config(self, config: Union[dict, str]) -> Union[dict, str]:
        if isinstance(config, dict):
            mlflow.log_dict(config, 'config.yaml')
        else:
            mlflow.log_artifact(config)
        return config
    
    def add_tags(self, tags:dict, rewrite=False):
        """
        If tag name already exists, it will be rewritten
        """
        mlflow.set_tags(tags)

    def set_status(self, status:str):
        """
        RUNNING, SCHEDULED, FINISHED, FAILED, KILLED
        """
        valid_statuses = [
            "RUNNING", 
            "SCHEDULED", 
            "FINISHED", 
            "FAILED", 
            "KILLED"
        ]
        if not status in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        mlflow.end_run(status)

    def set_iterations(self, iterations: int):
        tag = 'epochs' if self.train else 'steps'
        mlflow.set_tags({tag: iterations})

    def log_iteration(self, iteration:int):
        name = 'current_epoch' if self.train else 'step'
        mlflow.set_tags({name: iteration})


if __name__ == '__main__':
    params = {
        'key_token': 'R5V25ELMF8K44HN6ZDP8',
        'secret_token': 'QJPm5a8Ef4L7Ag5QRuemh93RFEXHSdXaxnkwhM0dRwa0mtf9qE',
        'subproject': True  # experiment format
    }
    experiment = 'experiment'
    run_name = 'train-debug'
    manager = ClearMLManager(**params, experiment='OCR/decoder-with-lm', run_name='example')

    manager.log_config('config.yaml')
    tags = {
        "current_epoch": '1'
    }
    manager.add_tags(tags)

    # ...

    tags = {
        "current_epoch": 2
    }
    manager.add_tags(tags)
