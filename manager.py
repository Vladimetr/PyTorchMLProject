import mlflow
from typing import Union


class BaseManager:
    def __init__(self):
        pass

    def log_step_metrics(self, metrics:dict, step:int):
        """
        Log step metrics for plot
        """
        pass

    def log_hyperparams(self, hparams:dict):
        pass

    def log_summary_metrics(self, metrics:dict):
        """
        Log average or best metrics
        """
        pass

    def log_config(self, config:Union[dict, str], name='config.yaml'):
        """
        Log config dict or config.yaml
        """
        pass


class MLFlowManager(BaseManager):
    def __init__(self, url:str, experiment='noname', 
                run_name:str="noname", tags:dict={}):
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
        run = mlflow.start_run(
            experiment_id=self.exp_id, 
            run_id=run_id, 
            run_name=run_name, 
            tags=tags, 
            description="my example"
        )
        self.run_id = run.info.run_id
        self.max_step = 0

    def log_step_metrics(self, metrics:dict, step:int):
        mlflow.log_metrics(metrics, step=step)
        self.max_step = max(self.max_step, step)

    def log_hyperparams(self, hparams:dict):
        mlflow.log_params(hparams)

    def log_summary_metrics(self, metrics:dict):
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
