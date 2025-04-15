from clearml import Task, Logger, TaskTypes, Dataset
from typing import Union, List, Optional
import os
import torch
import numpy as np
from sklearn import metrics
from .. import PROJECT_NAME
from .. import utils

"""
NOTE
in case of failed Dataset.get(dataset_id=ds_id) -> Connection refused

/home/v.kochetkov/.local/lib/python3.8/site-packages/clearml/datasets/dataset.py
#line:1637
old URL must be replaced with new one

I haven't found any alternatives
"""

matrix = Union[np.ndarray, torch.Tensor]

DATA_MOUNTS = ("/mnt/raid10/datasets/projects/noise_classification/", "/data/")

class ClearMLManager:
    def __init__(self, 
                 subproject:bool=False,  # experiment format
                 experiment='noname', 
                 run_name:str="noname",
                 train:bool=True, 
                 tags:List[str]=[],
                 resume:bool=False):
        """
        key_token, secret_token (str): authorization data. Get this from UI
        subproject (bool): format of experiment naming
        train (bool): type of experiment
        resume (bool): whether to resume given run_name.
            If False and run_name exists, overwrite it
        """
        # credentials
        key_token = "R5V25ELMF8K44HN6ZDP8"  #os.getenv("CLEARML_KEY_TOKEN")
        secret_token = "QJPm5a8Ef4L7Ag5QRuemh93RFEXHSdXaxnkwhM0dRwa0mtf9qE" # os.getenv("CLEARML_SEC_TOKEN")
        if not (key_token and secret_token):
            raise Exception('"CLEARML_KEY_TOKEN" and "CLEARML_SEC_TOKEN" '\
                            'must be defined in env')
        Task.set_credentials(key=key_token, secret=secret_token)
        
        task_type = TaskTypes.training if train else TaskTypes.testing
        if subproject:
            project_name = PROJECT_NAME + '/' + experiment
            task_name = run_name
        else:
            project_name = PROJECT_NAME
            task_name = experiment + '_' + run_name

        tasks = Task.get_tasks(project_name=project_name, task_name=task_name, 
                               allow_archived=False)
        if tasks and resume:
            assert len(tasks) == 1, f"Multiple tasks {tasks}"
            # self.task = tasks[0]
            self.task = Task.init(project_name=project_name, 
                                 task_name=task_name, 
                                 task_type=task_type,
                                 auto_connect_frameworks=False,
                                 auto_resource_monitoring=False, 
                                 auto_connect_streams=True,
                                 continue_last_task=tasks[0].task_id)
            # iteration will be continued
        else:
            # New task
            self.task = Task.init(project_name=project_name, 
                                task_name=task_name, 
                                task_type=task_type,
                                auto_connect_frameworks=False,
                                auto_resource_monitoring=False, 
                                auto_connect_streams=True)
            self.task.rename(task_name)  # if it was renamed
            self.task.set_initial_iteration(1)
            self._set_docker_info()

        # Turn off auto saveing ML models and other artifacts
        self.logger = Logger.current_logger()
        self.max_step = 0
        self.add_tags(tags)
        self.resume = resume
        print(f"ClearML experiment: '{project_name}/{task_name}'")

    def _set_docker_info(self):
        """
        These info is used for running in ClearML agent
        """
        docker_args = "-v|/mnt:/mnt|-v|/mnt/nvme/vovik/noise_classification/:/app/|-v|/mnt/raid10/datasets/projects/noise_classification/:/data|-w|/app/|--user|1018:1018" # os.getenv("DOC_ARGS")
        docker_image = "noise-cls:agent"  # os.getenv("AGENT_DOCIMAGE")
        self.task.set_base_docker(
            docker_arguments=docker_args.replace("|", " "),
            docker_image=docker_image,
        )

    def _validate_conf_matrix(self, conf_matrix:matrix, 
                              classes:List[str]):
        # validate confusion matrix
        n_classes = len(classes)
        shape = conf_matrix.shape
        if not (len(shape) == 2 and shape[0] == shape[1] == n_classes):
            raise ValueError('Mismatch matrix shape and N classes')

    def set_iterations(self, iterations:int):
        """
        Set number of train epochs
        or test steps (number of test batches)
        """
        self.n_iterations = iterations
        pass

    def log_iteration(self, iteration:int):
        """
        Log current epoch or test step
        """
        assert iteration <= self.n_iterations
        progress = int(iteration / self.n_iterations * 100)
        # persent 0..100
        self.task.set_progress(progress)

    def log_hyperparams(self, hparams: dict) -> dict:
        """ Hyperparams are defined in config manager:hparams
        NOTE: It's allowed to update hparams in UI. 
        So this method returns updated hparams to set them
        in train.py
        """
        if self.resume:
            # already logged
            return hparams
        hparams = self.task.connect(hparams, name='hparams')
        return hparams

    def log_config(self, config: Union[dict, str]) -> Union[dict, str]:
        """
        Log config dict or config.yaml
        NOTE: It's allowed to update config in UI. 
        So this method returns updated config to set it
        in train.py
        """
        if self.resume:
            # already logged
            return config
        config = self.task.connect_configuration(config, 
                                                 name='config.yaml')
        return config
    
    def log_metadata(self, metadata:dict):
        """ Additional static (unchangable) metadata like
        - experiment storage
        - train steps
        """
        if self.resume:
            # already logged
            return
        
        self.task.set_configuration_object(name='meta.yaml', 
                                           config_dict=metadata)

    def log_step_metrics(self, metrics: dict, step: int, prefix:str=None):
        """
        Log epoch metrics or test step metrics for plot
        NOTE: Don't log a lot of metrics (more than 200 per experiment),
        because it can slow down UI
        NOTE: Loss can be stored in metrics dict with key "*Loss"
        (for ex. "CrossEntropyLoss")
        NOTE: initial step must be 0 
        for either new experiment or resumed one
        Clearml append new values itself.
        Args:
            step (int): initital step is 0
        """
        if not self.resume:
            step += 1
        for metric_name, value in metrics.items():
            title = 'Metrics'
            if 'Loss' in metric_name:
                title = 'Loss'
                metric_name = metric_name.replace('Loss', '')
            if prefix:
                metric_name = prefix + "_" + metric_name
            self.logger.report_scalar(
                title=title, 
                series=metric_name, 
                value=value, 
                iteration=step
        )
        self.max_step = max(self.max_step, step)
            
    def log_summary_metrics(self, metrics: dict):
        """
        Log average or best metrics
        """
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
                             normalize=False):
        """
        Some managers support logging confusion matrix
        NOTE: xaxis="target" and yaxis="predict"
        """
        if not classes:
            n_classes = conf_matrix.shape[0]
            classes = list(map(str, range(n_classes)))
        # validate input
        self._validate_conf_matrix(conf_matrix, classes)
        if isinstance(conf_matrix, torch.Tensor):
            conf_matrix = conf_matrix.numpy()
        self.logger.report_confusion_matrix(
                    title, "ignored", 
                    iteration=self.max_step, matrix=conf_matrix,
                    xlabels=classes, ylabels=classes,
                    xaxis="target", yaxis="predict"
        )

        if not normalize:
            return
        
        # normalize by: row (pred), column(targ), all (pred&targ)
        norm_by = (
            ("row", "pred"),
            ("col", "targ"),
            # ("all", "pred & targ")
        )
        for by, name in norm_by:
            normed = utils.normalize_confusion_matrix(conf_matrix, by)
            self.logger.report_confusion_matrix(
                        f"{title} normed by {name}", 
                        "ignored", 
                        iteration=self.max_step, matrix=normed,
                        xlabels=classes, ylabels=classes,
                        xaxis="target", yaxis="predict"
            )

    def plot_roc_curve(self, 
                  preds:Union[torch.Tensor, np.ndarray], 
                  targs:Union[torch.Tensor, np.ndarray], 
                  pos_label:int=1,
                  best_criteria:str='g-mean',
                  class_name:str=None) -> float:
        """ Plot ROC curve and returns AUC
        Args:
            preds (Tensor, np.ndarray): (N, ) float (probs) (0..1)
            targs (Tensor, np.ndarray): (N, ) float (class labels)
            pos_label (int): positive class index
            best_criteria (str): 'g-mean'
        Returns:
            float: AUC value
        """
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        if not isinstance(targs, np.ndarray):
            targs = targs.numpy()
        fpr, tpr, threshs = metrics.roc_curve(targs, preds, 
                                            pos_label=pos_label)
        # (M, )
        threshs = np.where(threshs <= 1, threshs, 1)
        auc_score = metrics.auc(fpr, tpr)
        data = np.stack((fpr, tpr), axis=1)
        # plot
        self.logger.report_scatter2d(
            title=f"ROC curve {class_name or ''}",
            series='line',
            scatter=data,
            xaxis="FPR", yaxis="TPR (Recall)",
            labels=threshs.tolist()
        )
        if best_criteria == 'g-mean':
            g_means = np.sqrt(tpr * (1 - fpr))  # (M, )
            ix = np.argmax(g_means)
            value = g_means[ix]
        else:
            raise ValueError(f"Unknown best criteria '{best_criteria}'")
        
        # Plot best point
        best_values = \
            'thres={:.2f} auc={:.2f} {}={:.2f}'\
            .format(threshs[ix], auc_score, best_criteria, value)
        self.logger.report_scatter2d(
            title=f"ROC curve {class_name or ''}",
            series="best",
            scatter=data[ix:ix+1],
            xaxis="FPR", yaxis="TPR (Recall)",
            labels=[best_values],
            mode='markers'
        )
        return auc_score

    def plot_pr_curve(self,
                 preds:Union[torch.Tensor, np.ndarray], 
                 targs:Union[torch.Tensor, np.ndarray], 
                 pos_label:int=1,
                 best_criteria:str='f1',
                 class_name:str=None) -> float:
        """ Plot Precision-Recall curve and returns AUC
        Args:
            preds (Tensor, np.ndarray): (N, ) float (probs)
            targs (Tensor, np.ndarray): (N, ) float (class labels)
            pos_label (int): positive class index
            best_criteria (str): 'f1'
        Returns:
            float: AUC value
        """
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        if not isinstance(targs, np.ndarray):
            targs = targs.numpy()
        precs, recalls, threshs = \
            metrics.precision_recall_curve(targs, preds, pos_label=pos_label)
        precs, recalls = precs[ :-1], recalls[ :-1]
        corrects = recalls * precs > 0  # bool
        precs, recalls = precs[corrects], recalls[corrects], 
        threshs = threshs[corrects]
        assert np.all((0 <= threshs) & (threshs <= 1))
        data = np.stack((recalls, precs), axis=1)
        # (M, 2)
        # NOTE: xaxis=recalls, yaxis=precisions 
        auc_score = metrics.auc(recalls, precs)

        # plot
        self.logger.report_scatter2d(
            title=f"PR curve {class_name or ''}",
            series='line',
            scatter=data,
            xaxis="Recall", yaxis="Precision",
            labels=threshs.tolist()
        )
        if best_criteria == 'f1':
            f1_scores = 2 * precs * recalls / (precs + recalls)  # (M, )
            ix = np.argmax(f1_scores)
            value = f1_scores[ix]
        else:
            raise ValueError(f"Unknown best criteria '{best_criteria}'")
        
        # Plot best point
        best_values = \
            'thres={:.2f} auc={:.2f} {}={:.2f}'\
            .format(threshs[ix], auc_score, best_criteria, value)
        self.logger.report_scatter2d(
            title=f"PR curve {class_name or ''}",
            series='best',
            scatter=data[ix:ix+1],
            xaxis="Recall", yaxis="Precision",
            labels=[best_values],
            mode='markers'
        )
        return auc_score


class ClearMLDataset:
    """ Don't forget to do commit after uploading 
    plots, files, tables, etc
    """
    __project_name = 'Datasets/' + PROJECT_NAME
    __dataset_name = PROJECT_NAME.lower()

    def __init__(self, dataset:Dataset):
        """ Don't use this directly. User 'create' or 'get' instead """
        self.dataset = dataset
        self.logger = dataset.get_logger()
        self.id = dataset.id
        self.version = dataset._dataset_version

    @classmethod
    def create(cls, data:str,
               version:str=None,
               previous:Optional[List[str]]=None,
               tags:Optional[List[str]]=[],
               description:Optional[str]=None,
               sample_dur:Optional[float]=None,               
               data_size:Optional[int]=None,
               n_classes:Optional[int]=None,
               **kwargs):
        """
        Create new dataset with given version
        Args:
            data (str): /path/to/data.v{X}.{Y}-{postfix}.csv
            version (str): version in format like '1.2-test' 
                           or '1.0-mnist'
                           If None, version will be automatically parsed
                           from data_path
            previous (list[str]): parent IDs
            description (str): may refer to transformation 
                for example 'balancing'
            sample_dur (float): duration in sec of one sample (raw)
            data_size (int): number of samples in dataset. If defined, 
                add tag like "123k"
            n_classes (int): number of classes in dataset. If defined, 
                add tag like "50cls"
        Returns:
            ClearmlDataset
        """
        if previous and not isinstance(previous, list):
            raise TypeError("'previous' must be list of IDs")
        if not version:
            # get version from data path
            version = utils.get_version_from_path(data)

        # tags
        tags = tags or []
        if data_size:
            # TODO expand to '', 'k', 'm'
            tags.append(str(round(data_size / 1000)) + "k")
        if n_classes:
            tags.append(str(n_classes) + "cls")
        if sample_dur:
            tags.append(str(round(sample_dur, 1)) + "sec")

        # to connect git info task needs to be defined
        Task.init(project_name=cls.__project_name, 
                  task_name=cls.__dataset_name, 
                  task_type=Task.TaskTypes.data_processing,
                  auto_connect_frameworks=False,
                  auto_resource_monitoring=False, 
                  auto_connect_streams=False)
        dataset = Dataset.create(
                    parent_datasets=previous,
                    dataset_version=version,
                    description=description,
                    dataset_tags=tags,
                    use_current_task=True)
        dataset._task.set_user_properties(data=data, 
                                          data_size=data_size, 
                                          n_classes=n_classes, 
                                          sample_dur=sample_dur, 
                                          **kwargs)
        return cls(dataset)

    @classmethod
    def get(cls, version:Optional[str]="latest", id:Optional[str]=None):
        """
        Get dataset with given version or given ID
        Args:
            version (str): version. If "latest" get newest version
            id (str): ID
            NOTE: use either 'version' or 'id'
        Returns:
            ClearmlDataset
            None: if not found
        """
        if bool(version) == bool(id):
            raise ValueError("Define either 'version' or 'id'")
        
        if version == "latest":
            dataset = Dataset.list_datasets(cls.__project_name, 
                                            cls.__dataset_name)[-1]
            return cls(dataset)
        
        try:
            dataset = Dataset.get(dataset_id=id, 
                                dataset_version=version, 
                                dataset_project=cls.__project_name,
                                dataset_name=cls.__dataset_name,
                                only_completed=True)
        except ValueError:
            return
        return cls(dataset)
    
    @classmethod
    def get_by_data(cls, data:str):
        """
        Get dataset with given version or given ID
        Args:
            data (str): /path/to/data
        NOTE: data is one of dataset property (key "data")
        Returns:
            ClearmlDataset
            None: if not found
        """
        data1 = data
        # mounted dir
        if data.startswith(DATA_MOUNTS[0]):
            data2 = data.replace(DATA_MOUNTS[0], DATA_MOUNTS[1])
        else:
            data2 = data.replace(DATA_MOUNTS[1], DATA_MOUNTS[0])

        datasets: List[dict] = Dataset.list_datasets(cls.__project_name, 
                                                     cls.__dataset_name)
        for ds in datasets:
            ds_id = ds["id"]
            dataset = Dataset.get(dataset_id=ds_id)
            ds_data = dataset._task.get_user_properties()["data"]["value"]
            if data1 == ds_data:
                return cls(dataset)
            if data2 == ds_data:
                return cls(dataset)
        # does not exist
        return

    @classmethod
    def get_all(cls):
        """
        Get all datasets in given project
        Returns:
            list[str]: list of Dataset ID
        """
        datasets = Dataset.list_datasets(cls.__project_name, 
                                         cls.__dataset_name)
        ids = [ds["id"] for ds in datasets]
        return ids
    
    def add_tags(self, tags:List[str]):
        self.dataset._task.add_tags(tags)

    def add_metadata(self, datasize:int, **kwargs):
        """
        Metadata are visualized in 
        ClearML Task -> configuration -> properties
        It may refer to transformation info, like clean method 
        Args:
            datasize (int): number of examples - neccessary param
            key1: value
            key2: {"value": value, "description": "example"}
        """
        self.dataset._task.set_user_properties(datasize=datasize, **kwargs)

    def get_metadata(self, key:Optional[str]=None):
        """
        Args:
            key (str, None): if defined return only value of this key
                Otherwise return all metadata (dict)
        """
        metadata = self.dataset._task.get_user_properties()
        # drop unneccessary fields
        for k, data in metadata.items():
            metadata[k] = data["value"]
        if not key:
            return metadata
        return metadata[key]
    
    def get_data(self) -> str:
        data = self.get_data("data")
        return data
    
    def add_text(self, text:str, print=False):
        self.logger.report_text(text, print_console=print)

    def add_classes(self, classes:Union[str, List[str]]):
        if not isinstance(classes, str):
            fpath = "/tmp/classes.txt"
            with open(fpath, 'w') as f:
                for c in classes:
                    f.write(str(c) + '\n')
            classes = fpath
        self.dataset._task.connect_configuration(classes, name="classes")

    def upload_file(self, fpath:str, name:str):
        self.dataset._task.connect_configuration(fpath, name=name)    

    def upload_dict(self, data:dict, name:str):
        self.dataset._task.connect_configuration(data, name=name)    

    def add_histogram(self, data, bins:int,
                      name:str="Histogram",
                      series:str='data', 
                      xtitle:str='x',
                      ytitle:str='count'):
        """
        Plot histogram from given data
        Args:
            data (np.ndarray): 1-D array
            bins (int): number of columns with equal range
            name (str): name of histogram
            series (str): name of series
            NOTE: multiple histograms with same 'name'
            are plotted together with different 'series'
        """
        bins = list(range(20)) + [1000]
        counts, ranges = np.histogram(data, bins)
        ranges = list(map(str, ranges[ :-1]))
        self.logger.report_histogram(name,
                                     series,
                                     values=counts,
                                     xaxis=xtitle,
                                     yaxis=ytitle,
                                     xlabels=ranges)
        
    def add_bar(self, data, 
                      name:str='Bar', 
                      series:str='data', 
                      xtitle:str='x',
                      ytitle:str='y',
                      labels=None):
        """
        Plot bar for each value in array
        NOTE: number of columns = number of values in array
        For example [1, 3, 2, 5]
        - 1st column has height 1
        - 2nd column has height 3
        - 3rd column has height 2
        - 4th column has height 5
        Args:
            data (np.ndarray): 1-D array
            name (str): name of histogram
            series (str): name of series
            NOTE: multiple histograms with same 'name'
            are plotted together with different 'series'
        """
        self.logger.report_histogram(name,
                                     series,
                                     values=data,
                                     xaxis=xtitle,
                                     yaxis=ytitle,
                                     xlabels=labels)

    def add_pd_table(self, data,
                     name:str='Table', 
                     series:str='data',
                     extra_layout=None):
        """
        Add table from pandas DataFrame
        Args:
            data (pd.DataFrame): data table
            name (str): name of table
            series (str): name of series
            extra_layout (str): additional config for table
                See https://plotly.com/javascript/reference/layout/
        """
        self.logger.report_table(
                title=name,
                series=series,
                table_plot=data,
                extra_layout=extra_layout
        )

    def add_csv_table(self, csv_path:str,
                     name:str='Table', 
                     series:str='data',
                     extra_layout=None):
        """
        Add table from CSV file
        Args:
            csv_path (str): /path/to/table.csv
            name (str): name of table
            series (str): name of series
            extra_layout (str): additional config for table
                See https://plotly.com/javascript/reference/layout/
        """
        self.logger.report_table(
                title=name,
                series=series,
                csv=csv_path,
                extra_layout=extra_layout
        )

    def add_example(self, fpath:str, 
                    name:Optional[str]='example'):
        """
        Add some media file as an example. It can be viewed in
        Task information -> Debug samples
        NOTE: Don't add large files
        Args:
            fpath (str): /path/to/file
        """
        self.logger.report_media(title='examples', 
                                 series=name,
                                 local_path=fpath,
                                 stream=None,
                                 delete_after_upload=False)

    def commit(self):
        self.dataset.finalize()
        self.dataset._task.close()
        print("Dataset is logged to ClearML.", flush=True)


if __name__ == '__main__':
    task = Task.get_task("46d348ee509a499a88a1a5021c8500bf")

    print(type(task.get_status()))
    print(task.get_parameters_as_dict())

    exit()
    
    
    params = {
        'subproject': True  # experiment format
    }


    experiment = 'vova'
    run_name = 'test'
    resume = False
    manager = ClearMLManager(**params, experiment=experiment, run_name=run_name, resume=resume)
    a = manager.logger.get_flush_period()
    print(a)
    exit()


    manager.log_config("classes/join-classes_arca23.v2.json")

    # conf_matrix = np.load("/mnt/nvme/vovik/noise_classification/conf_matrix.npy")

    # manager.log_confusion_matrix(conf_matrix, normalize=True)

    exit()


    
    values = [3.4, 3.2]
    steps = [0, 1]
    for v, step in zip(values, steps):
        metrics_ = {
            "loss": v
        }
        manager.log_step_metrics(metrics_, step=step)

    exit()


    # pr, rec = 0.8932, 0.9260
    # manager.log_summary_metrics({
    #     "accuracy": 0.9754,
    #     "precision": pr,
    #     "recall": rec,
    #     "f1": 2 * pr * rec / (pr + rec)
    # })

    # data = simulate_func(k=5, noise=0.1)
    # for i in range(data.shape[0]):
    #     _, y = tuple(data[i].tolist())
    #     manager.log_step_metrics({"CrossEntropyLoss": y}, step=i*10)

    # data = simulate_func2(k=100, noise=0.0001)
    # for i in range(data.shape[0]):
    #     _, y = tuple(data[i].tolist())
    #     manager.log_step_metrics({"Precision": y}, step=i*10)

    # data = simulate_func2(k=200, noise=0.0002)
    # for i in range(data.shape[0]):
    #     _, y = tuple(data[i].tolist())
    #     manager.log_step_metrics({"Recall": y}, step=i*10)


    # data = np.asarray([45321, 34650])
    # manager.logger.report_histogram("Class distribution",
    #                                 "train data",
    #                                 values=data,
    #                                 xaxis='classes',
    #                                 yaxis='count')
    
    # data = np.asarray([5798, 4089])
    # manager.logger.report_histogram("Class distribution",
    #                                 "test data",
    #                                 values=data,
    #                                 xaxis='classes',
    #                                 yaxis='count')

    # image_open = Image.open('/mnt/nvme/vovik/tutorial/torch_project/data/1.jpg')
    # manager.logger.report_image(
    #         "cat", 
    #         "1", 
    #         iteration=1, 
    #         image=image_open
    #     )
    
    # image_open = Image.open('/mnt/nvme/vovik/tutorial/torch_project/data/2.jpg')
    # manager.logger.report_image(
    #         "cat", 
    #         "2", 
    #         iteration=1, 
    #         image=image_open
    #     )

    # image_open = Image.open('/mnt/nvme/vovik/tutorial/torch_project/data/3.jpg')
    # manager.logger.report_image(
    #         "dog", 
    #         "1", 
    #         iteration=1, 
    #         image=image_open
    #     )

    manager.log_config('config.yaml')
    tags = ['cnn', 'adadelta']
    manager.add_tags(tags)

    manager.log_hyperparams({
        "learning_rate": 0.0001,
        "optimizer": "adadelta",
        "weight_decay": 0.00001,
        "grad_norm": 2.0,
        "cnn_layers": 9
    })

    manager.log_metadata({
        "train_data": "data/processed/train_data.v1.csv",
        "test_data": "data/processed/test_data.v1.csv",
        "test_bs": 200,
        "train_bs": 500,
        "train_steps": 50000,
        "test_steps": 1200
    })


    # ...

    tags = ['gru', 'sgd']
    manager.add_tags(tags)

    exit()

    # ROC and PR curves
    B, C = 10, 2  # datasize and n_classes
    pred = np.random.rand(B, C)
    targ = np.random.randint(0, 2, (B, C)).astype(np.float32)

    for i in range(1000000):
        print(i)

    pred = np.reshape(pred, -1)
    targ = np.reshape(targ, -1)
    manager.plot_pr_curve(pred, targ)
    manager.plot_roc_curve(pred, targ)
