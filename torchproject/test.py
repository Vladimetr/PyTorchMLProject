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
from .data import CudaDataLoader, BucketingSampler, MyDataset
from . import utils
from .manager import MLFlowManager, BaseManager
from .models import model_init
from .metrics import init_loss, BinClassificationMetrics

EXPERIMENTS_DIR = 'dev/experiments'
TB_LOGS_DIR = 'dev/tensorboard/logs'
manager = None


class ConfigError(Exception):
    pass

def get_new_run_id(runs_dir:str) -> int:
    existed_runs = os.listdir(runs_dir)
    max_id = 0
    if existed_runs:
        max_id = max([int(run_name[:3]) \
                      for run_name in existed_runs])
    return max_id + 1


def log_metrics(metrics:dict, step:int, logger,
                manager:BaseManager=None,
                tb_writer:SummaryWriter=None):

    # log metrics to .csv
    log_line = [str(v) for _, v in metrics.items()]
    logger.info(' '.join(log_line))

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
        loss,
        metrics_computer:BinClassificationMetrics,
        ) -> dict:
    x, target = batch
    with torch.no_grad():
        logits, probs = model(x)
    # logits - before activation (for loss)
    # probs - after activation   (for acc)

    # CrossEntropy loss
    output = loss(logits, target.float())  # is graph (for backward)
    loss_value = output.item()     # is float32

    # Check if loss is nan
    if torch.isnan(output) or isnan(loss_value):
        message = f"Loss is NaN"
        raise Exception(message)

    # Metrics computing
    metrics = {"loss": loss_value}
    metrics_ = metrics_computer.compute(probs, target,
                                       accumulate=True)
    metrics.update(metrics_)
    return metrics


@status_handler
def main(data:str,
         config:Union[str, dict]='config.yaml',
         batch_size:int=500,
         experiment:str='experiment',
         use_manager:bool=True,
         tensorboard:bool=False,
         data_shuffle:bool=True,
         log_step:int=1,
         comment:str=None,
    ):
    """
    :param model_dir: куда сохранять результаты обучения (при debug_mode=False)
                        if None, model_dir = date_time
    :param params: dict with train and feature params.
                    if params == 'default' take params from params.py
    :param data_dir: dir with: npy/ , data.csv
    :param retrain: path/to/model.pt that we need to re-train
    :param train_steps: сколько батчей прогонять в каждой эпохи
                    if None, all batches
    :param test_steps: сколько тестовых батчей прогонять после каждой эпохи
                        if None, all Test Set
    :param debug_mode: if True, without save model, summary and logs
    """
    global manager
    if isinstance(config, str):
        # load config from yaml
        config = utils.config_from_yaml(config)

    test_params = config["test"]
    data_params = config["data"]
    model_params = config["model"]
    manager_params = config["manager"]

    runs_dir = os.path.join(EXPERIMENTS_DIR, 
                            experiment.replace(' ', '_'), 
                            'test')
    if not osp.exists(runs_dir):
        os.makedirs(runs_dir)
    run_name = '{:03d}'.format(get_new_run_id(runs_dir))
    if comment:
        run_name += '_' + comment
    # init dirs
    run_dir = osp.join(runs_dir, run_name)
    os.makedirs(run_dir)
    os.makedirs(osp.join(run_dir, 'weights/'))
    # copy config
    utils.dict2yaml(config, osp.join(run_dir, 'config.yaml'))
    # create metadata of this experiment
    metadata = {
        "data": data,
        "batch_size": batch_size,

    }
    utils.dict2yaml(metadata, osp.join(run_dir, 'meta.yaml'))
    # init files for log metrics
    test_logfile = osp.join(run_dir, 'test.csv')
    test_logger = utils.get_logger('test', test_logfile)
    print(f"Experiment storage: '{run_dir}'")

    # Init manager
    if use_manager:
        manager = MLFlowManager(
            url=manager_params["url"],
            experiment=experiment,
            run_name='test-' + run_name,
            tags={'mode': 'test'}
        )
        manager.log_hyperparams(manager_params["hparams"])
        manager.log_file("config.yaml")
        manager.log_dict(metadata, 'meta.yaml')
        print(f"Manager experiment run name: {'test-' + run_name}")

    # Load test data
    test_set = MyDataset(data, params=data_params)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(test_set, collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)

    # Define model
    model_name = test_params["model"]
    model_params = config["model"][model_name]
    if not model_params["weights"]:
        raise ConfigError("Weights are not defined")
    model = model_init(model_name,
                       model_params,
                       train=True,
                       device="cuda:0")

    # Tensorboard writer
    if tensorboard:
        log_dir = osp.join(TB_LOGS_DIR, experiment, 'test', run_name)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logs: '{log_dir}'")
    else:
        writer = None

    # Define loss
    loss_name = test_params["loss"]
    loss_params = config["loss"][loss_name]
    loss = init_loss(loss_name, loss_params, device="cuda:0")

    # Init test metrics computer
    compute_metrics = test_params["metrics"]
    metrics_computer = BinClassificationMetrics(
                               compute_metrics=compute_metrics)
    title = ' '.join([name for name in \
                      ["Loss"] + compute_metrics])
    test_logger.info(title)

    test_set.shuffle(15)
    for step, batch in enumerate(test_set):
        metrics = test_step(
            model,
            batch,
            loss,
            metrics_computer
        )
        if step % log_step == 0:
            log_metrics(metrics, step=step, 
                        logger=test_logger,
                        manager=manager,
                        tb_writer=writer)

    avg_metrics = metrics_computer.summary()
    print("--- Average metrics ---")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")
    if manager:
       manager.log_summary_metrics(avg_metrics)

    if writer: writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lang classificator')
    parser.add_argument('--config', '-cfg', type=str, default='config.yaml', 
                        help='path/to/config.yaml')
    parser.add_argument('--data', '-d', type=str, 
                        default='data/test_manifest.csv',
                        help='path/to/data')
    parser.add_argument('--batch_size', '-bs', type=int, default=20)
    parser.add_argument('--experiment', '-exp', default='experiment', 
                    help='Name of existed MLFlow experiment')
    parser.add_argument('--manager', '-mng', action='store_true', 
                        dest='use_manager', default=False, 
                        help='whether to use ML experiment manager')
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
