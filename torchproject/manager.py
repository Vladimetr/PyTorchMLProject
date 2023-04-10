import mlflow
from clearml import Task, Logger, TaskTypes
from typing import Union, List

PROJECT_NAME = 'MyProject'

class BaseManager:
    def __init__(self, train=True):
        """
        train (bool): mode of experiment (train or test)
        """
        pass

    def add_tags(self, tags:Union[List[str], dict]):
        """
        Tag might be str, like ['best']
        or dict, like {'mode': 'train'}
        """
        pass

    def log_step_metrics(self, metrics:dict, step:int):
        """
        Log step metrics for plot
        """
        pass

    def log_epoch_metrics(self, metrics:dict, epoch:int):
        """
        Log epoch metrics for plot
        """
        pass

    def log_hyperparams(self, hparams:dict):
        pass

    def log_summary_metrics(self, metrics:dict, type='best'):
        """
        Log average or best metrics
        """
        pass

    def log_config(self, config:Union[dict, str], name='config.yaml'):
        """
        Log config dict or config.yaml
        """
        pass

    def set_status(self, status:str):
        """
        Some managers support statuses, like
        finished or failed
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


class ClearMLManager(BaseManager):
    def __init__(self, key_token:str, secret_token:str, 
                 experiment='noname', 
                 run_name:str="noname", 
                 train:bool=True, 
                 tags:dict={}):
        Task.set_credentials(key=key_token, secret=secret_token)
        task_type = TaskTypes.training if train else TaskTypes.testing
        task_name = experiment + '_' + run_name
        self.task = Task.init(project_name=PROJECT_NAME, 
                              task_name=task_name, 
                              task_type=task_type)
        self.logger = Logger.current_logger()
        self.max_step = 0
        self.add_tags(tags)
        print(f"ClearML experiment: '{PROJECT_NAME}/{task_name}'")
        
    def log_hyperparams(self, hparams: dict):
        self.task.connect(hparams, name='hparams')

    def log_config(self, config: Union[dict, str], name='config.yaml'):
        if not isinstance(config, str):
            raise NotImplementedError()
        self.task.connect_configuration(config, name=name)

    def log_step_metrics(self, metrics: dict, step: int):
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title=metric_name, 
                series='step', 
                value=value, 
                iteration=step
        )
        self.max_step = max(self.max_step, step)
            
    def log_epoch_metrics(self, metrics: dict, epoch: int):
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title=metric_name, 
                series='epoch', 
                value=value, 
                iteration=epoch
        )
        self.max_step = max(self.max_step, epoch)

    def log_summary_metrics(self, metrics: dict, type='best'):
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title=metric_name, 
                series=type, 
                value=value, 
                iteration=self.max_step
        )
            
    def add_tags(self, tags:Union[List[str], dict]):
        if isinstance(tags, dict):
            tags = list(tags.values())
        self.task.add_tags(tags)
            

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

    def log_epoch_metrics(self, metrics:dict, epoch:int):
        self.log_step_metrics(metrics, step=epoch)

    def log_hyperparams(self, hparams:dict):
        mlflow.log_params(hparams)

    def log_summary_metrics(self, metrics:dict, type='best'):
        mlflow.log_metrics(metrics, step=self.max_step + 1)
    
    def log_dict(self, data:dict, fname:str):
        mlflow.log_dict(data, fname)

    def log_file(self, fpath:str, subdir:str=None):
        mlflow.log_artifact(fpath, subdir)

    def log_config(self, config: Union[dict, str], name='config.yaml'):
        if isinstance(config, dict):
            self.log_dict(config, name)
        else:
            self.log_file(config)

    def add_tags(self, tags:dict):
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
    url = 'http://192.168.11.181:3500'
    experiment = 'experiment'
    run_name = 'train-debug'
    manager = MLFlowManager(url, experiment, run_name)

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
