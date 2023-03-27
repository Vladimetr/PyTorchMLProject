import mlflow


class BaseManager:
    def __init__(self) -> None:
        pass


class MLFlowTrainManager(BaseManager):
    def __init__(self, url:str, experiment='noname', 
                run_name:str="noname"):
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
            tags={'mode': 'train'}, 
            description="my example"
        )
        self.run_id = run.info.run_id
        self.max_epoch = 0

    def log_epoch_metrics(self, metrics:dict, epoch:int):
        mlflow.log_metrics(metrics, step=epoch)
        self.max_epoch = max(self.max_epoch, epoch)

    def log_hyperparams(self, hparams:dict):
        mlflow.log_params(hparams)

    def log_summary_metrics(self, metrics:dict):
        mlflow.log_metrics(metrics, step=self.max_epoch + 1)
    
    def log_dict(self, data:dict, fpath:str):
        mlflow.log_dict(data, fpath)

    def log_file(self, fpath:str, subdir:str=None):
        mlflow.log_artifact(fpath, subdir)

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
    experiment = 'Example'
    id = 1
    manager = MLFlowTrainManager(url, experiment, id)

    manager.log_config('config.yaml')