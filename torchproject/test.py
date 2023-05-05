"""
Скрипт запуска модели на тестовой выборке,
вывод результатов метрик
"""

import torch
import os
import os.path as osp
from math import isnan
from typing import Union
import argparse
from torch.utils.tensorboard import SummaryWriter
from .models import Model
from .data import CudaDataLoader, BucketingSampler, AudioDataset
from . import utils
from .utils.manager import BaseManager, MLFlowManager, ClearMLManager
from .models import model_init
from .metrics import init_loss, BinClassificationMetrics, Loss
from .utils import EXPERIMENTS_DIR, TB_LOGS_DIR

manager = None


class ConfigError(Exception):
    pass

def get_train_run(experiment:str, run_id:int) -> Union[str, None]:
    """
    Get run name in given train experiment with given ID
    Raises:
        ValueError "Experiment {} doesn't exists"
        AssertionError: "Multiple runs with ID {run_id}: {run_names}"
    Returns:
        str: Run name. For ex. '002_comment'
        None: If given RunID doesn't exists
    """
    exp_dir = osp.join(EXPERIMENTS_DIR, experiment, 'train')
    try:
        run_names = os.listdir(exp_dir)
    except:
        raise ValueError(f"Experiment '{experiment}' doesn't exists")
    # get run names with given RunID
    run_names = list(filter(lambda x: int(x[ :3]) == run_id,
                            run_names))
    assert len(run_names) in [0, 1], \
            f"Multiple runs with ID {run_id}: {run_names}"
    if not run_names:
        return None
    return run_names[0]


def get_test_run(experiment:str, train_run_id:int=None) -> str:
    """
    Get test experiment run with reference to train RunID (opt)
    NOTE: if 'train_run_id' is not None, it must be existed
    Returns:
        str: new test run name
    """
    exp_dir = osp.join(EXPERIMENTS_DIR, experiment, 'test')
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)
    run_names = os.listdir(exp_dir)

    if train_run_id:
        prefix = '{:03d}-'.format(train_run_id)
    else:
        prefix = ''

    new_run_id = 1
    while prefix + str(new_run_id) in run_names:
        new_run_id += 1
    return prefix + str(new_run_id)
    

def log_metrics(metrics:dict, step:int, 
                metrics_computer=BinClassificationMetrics,
                manager:BaseManager=None,
                tb_writer:SummaryWriter=None):
    metrics_computer.log_metrics(metrics, step=step)
    # Manager
    if manager:
        manager.log_step_metrics(metrics, step)
    # Tensorboard
    if tb_writer:
        for metric_name, value in metrics.items():
            if metric_name == 'loss':
                tb_writer.add_scalar('Loss/CrossEntropy', value, 
                                     step)
            else:
                tb_writer.add_scalar('Metrics/' + metric_name, 
                                     value, step)

def status_handler(func):
    def run_train(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as error:
            # train is failed
            if manager is not None:
                manager.set_status("FAILED")
            raise error
        else:
            # train is successfull
            if manager is not None:
                manager.set_status("FINISHED")
    return run_train


def test_step(
        model:Model,
        batch:tuple,
        loss_computer:Loss,
        metrics_computer:BinClassificationMetrics,
        ) -> dict:
    x, target = batch
    with torch.no_grad():
        logits, probs = model(x)
    # logits - before activation (for loss)
    # probs - after activation   (for acc)

    # CrossEntropy loss
    loss, loss_values = loss_computer(logits, target)

    # Check if loss is nan
    if torch.isnan(loss) or \
        any([isnan(v) for v in loss_values.values()]):
        message = f"Loss is NaN"
        raise Exception(message)

    # Metrics computing
    metrics = metrics_computer.compute(probs, target,
                                       accumulate=True)
    metrics_computer.add_summary(loss_values)
    metrics.update(loss_values)
    return metrics


@status_handler
def main(data:str,
         config:Union[str, dict]='config.yaml',
         batch_size:int=500,
         gpu_id:int=0,
         no_save:bool=False,
         experiment:str='experiment',
         run_id:int=None,
         weights:str='best.pt',
         use_mlflow:bool=False,
         use_clearml:bool=False,
         tensorboard:bool=False,
         data_shuffle:bool=True,
         log_step:int=1,
         comment:str=None,
    ):
    """
    data(str): path/to/data
    config (str, dict): config dict or path/to/config.yaml
    experiment (str): experiment name
    run_id (int): train experiment RunID for reference, i.e.
        loading config and specific weights
    weights (str): weights name to load from given train run
    use_mlflow (bool): whether to manage experiment with MLFlow
    use_clearml (bool): whether to manage experiment with ClearML
    tensorboard (bool): whether to log step metrics to TB
    data_shuffle (bool): whether to shuffle data
    log_step (int): interval of loggoing step metrics
    comment (str): postfix for experiment run name
    """
    experiment = experiment.lower().replace(' ', '_')
    global manager
    if use_clearml and use_mlflow:
        raise ValueError("Choose either mlflow or clearml for management")
    writer, logger, run_dir = None, None, None
    hparams = dict()

    # Validate device
    num_valid_gpus = torch.cuda.device_count()
    if gpu_id >= num_valid_gpus:
        raise ValueError(f"Only {num_valid_gpus} GPUs are available")
    device = f"cuda:{gpu_id}"

    # Get reference to train RunID
    if run_id:
        train_run_name = get_train_run(experiment, run_id)
        if not train_run_name:
            raise ValueError(f"RunID {run_id} in experiment "
                             f"'{experiment}' doesn't exists")
        train_run_dir = osp.join(EXPERIMENTS_DIR, experiment,
                                 'train', train_run_name)
        config = osp.join(train_run_dir, 'config.yaml')
        weights = osp.join(train_run_dir, 'weights', weights)
    else:
        weights = None  # weights must be defined in config

    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config)
    else:
        config_yaml = '/tmp/config.yaml'
        config = dict(config)  # copy
    manager_params = config["manager"]

    if not no_save:
        # Define test Run name
        run_name = get_test_run(experiment, run_id)
        if comment:
            run_name += '_' + comment
            
        # Create storage
        run_dir = os.path.join(EXPERIMENTS_DIR, experiment,
                            'test', run_name)
        os.makedirs(run_dir)

        # Init manager
        if use_mlflow or use_clearml:
            params = {
                "experiment": experiment,
                "run_name": 'test-' + run_name,
                "train": False,
            }
            if use_clearml:
                params.update(manager_params["clearml"])
                manager = ClearMLManager(**params)
            else:
                params.update(manager_params["mlflow"])
                manager = MLFlowManager(**params)

            manager.log_hyperparams(manager_params["hparams"])
            # log and update config if it's defined in experiment
            config_yaml = manager.log_config(config_yaml)
            config = utils.config_from_yaml(config_yaml)
            hparams = config["manager"]["hparams"]
            # log and update if it was changed
            hparams = manager.log_hyperparams(hparams)  
            print(f"Manager experiment run name: {'test-' + run_name}")

        # save config
        config_yaml = osp.join(run_dir, 'config.yaml')
        utils.dict2yaml(config, config_yaml)
        # init files for log metrics
        logfile = osp.join(run_dir, 'test.csv')
        logger = utils.get_logger('test', logfile)
        print(f"Experiment storage: '{run_dir}'")

    # Load main params
    utils.update_given_keys(config, hparams)
    test_params = config["test"]
    model_name = config["model"]
    model_params = config["models"][model_name]
    preprocess_params = config["preprocess"]
    n_classes = config["n_classes"]

    # Load test data
    test_set = AudioDataset(data, n_classes=n_classes,
                            preprocess_params=preprocess_params)
    data_size = len(test_set)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(gpu_id, test_set, 
                              collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)
    test_steps = len(test_set)  # number of test batches

    # Redefine weights
    if weights:
        model_params["weights"] = weights

    # Add specific info
    if manager:
        manager.set_iterations(test_steps)
        weights_name = osp.split(model_params["weights"])[1]
        manager.add_tags({'weights': weights_name}, rewrite=True)

    # Define metadata
    metadata = {
            "data": data,
            "batch_size": batch_size,
            "data_size": data_size,
            "test_steps": test_steps,
            "storage": run_dir,
            "weights": model_params["weights"],
            
    }
    utils.pprint_dict(metadata)
    if not no_save:
        meta_yaml = osp.join(run_dir, 'meta.yaml')
        utils.dict2yaml(metadata, meta_yaml)
        if manager:
            manager.log_metadata(metadata)

    # Define model
    if not model_params["weights"]:
        raise ConfigError("Weights are not defined")
    model = model_init(model_name,
                       model_params,
                       train=True,
                       device=device)
    
    # Tensorboard writer
    if not no_save and tensorboard:
        log_dir = osp.join(TB_LOGS_DIR, experiment, 'test', run_name)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logs: '{log_dir}'")

    # Define loss
    loss_name = test_params["loss"]
    loss_params = config["loss"][loss_name]
    loss = init_loss(loss_name, loss_params, device=device)

    # Init test metrics computer
    compute_metrics = test_params["metrics"]
    metrics_computer = BinClassificationMetrics(
        step=True,
        n_classes=n_classes,
        pos_classes=config["pos_classes"],
        compute_metrics=compute_metrics,
        logger=logger
    )

    test_set.shuffle(15)
    for step, batch in enumerate(test_set):
        metrics = test_step(
            model,
            batch,
            loss,
            metrics_computer
        )
        metrics = {k: metrics[k] for k in test_params["metrics"]}
        if (step + 1) % log_step == 0:
            log_metrics(metrics, step=step+1, 
                        metrics_computer=metrics_computer,
                        manager=manager,
                        tb_writer=writer)

    sum_metrics = metrics_computer.summary()
    print("\n--- Summary metrics ---")
    for k in test_params["metrics"]:
        print(f"{k}: {sum_metrics[k]}")
    # Print summary conf matrix
    if n_classes > 2:
        metrics_computer.print_conf_matrix(sum_metrics["conf_matrix"])
    bin_conf_matrix = sum_metrics["bin_conf_matrix"]
    metrics_computer.print_conf_matrix(bin_conf_matrix)
    
    if manager:
       sum_metrics = {k: sum_metrics[k] for k in test_params["metrics"]}
       manager.log_summary_metrics(sum_metrics)
       manager.log_confusion_matrix(bin_conf_matrix)
       manager.close()

    if writer: writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lang classificator')
    parser.add_argument('--config', '-cfg', type=str, default='config.yaml', 
                        help='path/to/config.yaml')
    parser.add_argument('--data', '-d', type=str, 
                        default='data/processed/test_manifest.v1.csv',
                        help='path/to/data')
    parser.add_argument('--batch_size', '-bs', type=int, default=20)
    parser.add_argument('--gpu', type=int, dest="gpu_id", default=0,
                        help='which GPU to use')
    parser.add_argument('--no-save', '-ns', action='store_true', 
                        default=False, 
                        help='no save results')
    parser.add_argument('--experiment', '-exp', default='experiment', 
                    help='experiment name')
    parser.add_argument('--run-id', '-r', type=int, default=None,
                    help='train RunID for reference')
    parser.add_argument('--weights', '-w', type=str,
                        default='best.pt', 
                    help='Weights name for loading from this run')
    parser.add_argument('--mlflow', action='store_true', 
                        dest='use_mlflow', default=False, 
                        help='whether to use MLFlow for experiment manager')
    parser.add_argument('--clearml', action='store_true', 
                        dest='use_clearml', default=False, 
                        help='whether to use ClearML for experiment manager')
    parser.add_argument('--tensorboard', '-tb', action='store_true', 
                        default=False, 
                        help='whether to use Tensorboard')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    parser.add_argument('--comment', '-m', type=str, default=None, 
                    help='Postfix for experiment run name')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    main(**args)
