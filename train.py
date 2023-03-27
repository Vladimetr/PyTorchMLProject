"""
Какие модификации стоит внести
1) на тестах больше метрик: recall, precision, miss, false alarm ...
   нужно сравнивать метрики при разных значениях порога: [metrics(threhold(min)..metrics(threhold_max)]
2) сохранять модель той эпохи, на которой были тестовые метрики лучше, чем предыдущие
3) заменить logging на принты. Запускать контейнер -d без --rm
вывод всегда можно просмотреть docker logs container_ID &> training_process.log
 или
оставить как есть. Только результаты тестов и обучения логировать в разные файлы
4) импортироват либы в тех функциях, в которых они нужны
5) Cоздать единную папку logdir, где будут хранится результаты обучения всех моделей (кривые обучения и кривые тестов). Логи хранятся logdir/date_time(timeformat)
Добавить add_hparams в TensorBoard в конце тестирования. Лучшие результаты метрик по всем эпохам сохранять в hparams-metrics.
Тогда при запуске ТБ будут отображены таблица со всеми гиперпараметрами. Через фильтр Runs можно сравнивать только кривые тестов, оценивать лучшую кривую и сразу же через таблицы находить значения гиперпараметров.
"""
import torch
import os
import os.path as osp
from shutil import copyfile
from typing import Union
from math import isnan as math_isnan
from data import CudaDataLoader, BucketingSampler, MyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import argparse
import logging
import utils
from metrics import init_loss
from models import Model, model_init
from metrics import Metrics
# from test import test_step


EXPERIMENTS_DIR = 'dev/experiments'


def get_new_run_dir(runs_dir:str) -> str:
    try:
        existed_runs = os.listdir(runs_dir)
    except FileNotFoundError:
        existed_runs = []
    max_id = 0
    if existed_runs:
        max_id = max([int(run_name) for run_name in existed_runs])
    new_run_name = '{:03d}'.format(max_id + 1)
    new_run_dir = osp.join(runs_dir, new_run_name)
    return new_run_dir


def get_better_metrics(metrics1:dict, metrics2:dict) -> dict:
    """
    Here you need define best criteria
    Returns:
        dict: one of two metrics dict which is better
    """
    # For example
    if metrics1["loss"] < metrics2["loss2"]:
        return metrics1
    return metrics2


def train_step(
        model:Model,
        batch:tuple,
        optimizer,
        loss,
        metrics_computer:Metrics,
        train_params:dict,
        ) -> dict:
    # обнуление предыдущих градиентов
    optimizer.zero_grad()          

    x, target = batch
    # target - one hot (B, C)

    logits, probs = model(x)
    # logits - before activation (for loss)
    # probs - after activation   (for acc)

    # CrossEntropy loss
    output = loss(logits, target.float())  # is graph (for backward)
    loss_value = output.item()     # is float32

    # Check if loss is nan
    if torch.isnan(output) or math_isnan(loss_value):
        message = f"Loss is NaN"
        raise Exception(message)

    # обратное распр-е ошибки.
    # для каждого параметра модели w считает w.grad
    # здесь НЕ обновляются веса!
    output.backward()

    clip_grad_norm_(model.parameters(), train_params['grad_norm'])
    # prevent exploding gradient

    # здесь обновление весов
    # w_new = w_old - lr * w.grad
    optimizer.step()

    # check if grads are not NaN
    model.validate_grads()

    # metrics computing
    metrics = {"loss": loss_value}
    metrics_ = metrics_computer.compute(probs, target,
                                       accumulate=False)
    metrics.update(metrics_)
    return metrics


def log_metrics(metrics:dict, epoch:int, step:int, logger,
                tb_writer:SummaryWriter=None):

    # log metrics to .csv
    log_line = [epoch, step] + [v for _, v in metrics.items()]
    logger.info(' '.join([str(v) for v in log_line]))

    # Tensorboard
    if tb_writer:
        for metric_name, value in metrics.items():
            tb_writer.add_scalar(metric_name, 
                                 value, step)
        tb_writer.add_scalar('Loss', metrics["loss"], step)


def main(train_data:str,
         test_data:str,
         config:Union[str, dict]='config.yaml',
         epochs:int=15,
         batch_size:int=500,
         pretrained:str=None,
         experiment:str='noname',
         comment:str=None,
         debug:bool=True,
         tensorboard:bool=False,
         data_shuffle:bool=True,
         log_step:int=1,
    ):
    # model_dir=None, params='default', data_dir='data', epochs=15, batch_size=500,
    #       retrain=None, train_steps=None, test_steps=None, debug_mode=False):
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
    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config_yaml)
    else:
        raise NotImplementedError("Config dict is not implemented yet")
    train_params = config["train"]
    test_params = config["test"]
    data_params = config["data"]
    train_logfile, test_logfile = None, None

    # Create experiment run dir
    if not debug:
        runs_dir = os.path.join(EXPERIMENTS_DIR, experiment)
        run_dir = get_new_run_dir(runs_dir)
        if comment:
            run_dir += '_' + comment
        # init dirs
        os.makedirs(run_dir)
        os.makedirs(osp.join(run_dir, 'weights/'))
        # copy config
        copyfile(config_yaml, osp.join(run_dir, 'config.yaml'))
        # init files for log metrics
        train_logfile = osp.join(run_dir, 'train.log')
        test_logfile = osp.join(run_dir, 'test.log')
        print(f"Experiment storage: '{run_dir}'")

    # Files for logging metrics
    train_logger = utils.get_logger('train', train_logfile)
    test_logger = utils.get_logger('test', test_logfile)

    # Load train data
    train_set = MyDataset(train_data, params=data_params)
    sampler = BucketingSampler(train_set, batch_size, shuffle=data_shuffle)
    train_set = CudaDataLoader(train_set, collate_fn=train_set.collate, 
                               pin_memory=True, num_workers=4,
                               batch_sampler=sampler)

    # Load test data
    test_set = MyDataset(test_data, params=data_params)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(test_set, collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)
    
    # Define model
    model_name = train_params["model"]
    model_params = config["model"][model_name]
    model_params["weights"] = pretrained  # None of path/to/model.pt
    model = model_init(model_name,
                       model_params,
                       train=True,
                       device="cuda:0")

    # Define optimizer
    if train_params["opt"] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_params["learning_rate"], 
            weight_decay=train_params['weight_decay'])
    else:
        raise Exception('No optimizer: {}'.format(train_params["opt"]))

    # Tensorboard writer
    if not debug and tensorboard:
        log_dir = osp.join(run_dir, 'tb_logs')
        os.makedirs(log_dir)
        writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
        writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))
        print(f"Tensorboard logs: '{log_dir}'")
    else:
        writer_test = writer_train = None

    # Define loss
    loss_name = train_params["loss"]
    loss_params = config["loss"][loss_name]
    loss = init_loss(loss_name, loss_params, device="cuda:0")

    # Init train metrics computer
    compute_metrics = train_params["metrics"]
    train_metrics_computer = Metrics(n_classes=config["n_classes"],
                               compute_metrics=compute_metrics)
    title = ' '.join([name for name in \
                      ["Epoch", "Step", "Loss"] + compute_metrics])
    train_logger.info(title)

    # Init test metrics computer
    compute_metrics = test_params["metrics"]
    test_metrics_computer = Metrics(n_classes=config["n_classes"],
                               compute_metrics=compute_metrics)
    title = ' '.join([name for name in \
                      ["Epoch", "Step", "Loss"] + compute_metrics])
    test_logger.info(title)

    best_metrics = None
    for ep in range(1, epochs + 1):
        print(f"{ep}/{epochs} Epoch...")

        model.train()
        train_set.shuffle(ep)
        for i, batch in enumerate(train_set):
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss=loss,
                metrics_computer=train_metrics_computer,
                train_params=train_params,
            )
            if i % log_step == 0:
                log_metrics(metrics, epoch=ep, step=i, 
                            logger=train_logger,
                            tb_writer=writer_train)
                
        # Saving
        if not debug:
            weights_path = osp.join(run_dir, f"weights/{ep}.pt")
            model.save(weights_path)
            print(f"Weights save: '{weights_path}'")

        print('\n------------- Test ---------------')

        continue

        for i, batch in enumerate(test_set):
            metrics = test_step(
                model=model,
                batch=batch,
                loss=loss,
                metrics_computer=test_metrics_computer,
                step=test_iter,
                tb_writer=writer_test,
                log_step=log_step)
            
            log_line = [ep, i] + [v for _, v in metrics.items()]
            test_logger.info(' '.join(log_line))

            test_iter += 1
            
            
        print("--- Average metrics ---")
        metrics = test_metrics_computer.summary()
        for k, v in metrics:
            print(f"{k}: {v}")

        # Сheck whether it's the best metrics
        if best_metrics is None or \
                get_better_metrics(metrics, best_metrics) is metrics:
            best_metrics = metrics
            print('New best results')
            # save best weights
            if not debug:
                weights_path = osp.join(run_dir, f"weights/weights.pt")
                model.save(weights_path)
                print(f"Weights save: '{weights_path}'")

        test_metrics_computer.reset_summary()

        # ---- END OF EPOCH

    if not writer_train: writer_train.close()
    if not writer_test:  writer_test.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--train-data', type=str, 
                        default='data/train_manifest.csv')
    parser.add_argument('--test-data', type=str, 
                        default='data/test_manifest.csv')
    parser.add_argument('--batch-size', '-bs', type=int, 
                        default=100)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--debug', '-d', action='store_true', default=False, 
                        help='no save results')
    parser.add_argument('--pretrained', '-pr', default=None, 
                        help='path/to/pretrained/weigths.pt')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    main(**args)