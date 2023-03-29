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
import os
import os.path as osp
from typing import Union
from math import isnan as math_isnan
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from .data import CudaDataLoader, BucketingSampler, MyDataset
from . import utils
from .metrics import init_loss, BinClassificationMetrics
from .models import Model, model_init
from .manager import MLFlowManager
from .test import test_step
from .utils import EXPERIMENTS_DIR, TB_LOGS_DIR

manager = None


def get_new_run_id(runs_dir:str) -> int:
    existed_runs = os.listdir(runs_dir)
    max_id = 0
    if existed_runs:
        max_id = max([int(run_name[:3]) \
                      for run_name in existed_runs])
    return max_id + 1


def get_better_metrics(metrics1:dict, metrics2:dict) -> dict:
    """
    Here you need define best criteria
    Returns:
        dict: one of two metrics dict which is better
    """
    # For example
    if metrics1["TP"] > metrics2["TP"]:
        return metrics1
    return metrics2


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


def create_tb_writer(experiment:str, run_name:str, 
                     mode:str=None) -> SummaryWriter:
    """
    mode (str): 'train', 'test'
    """
    log_dir = osp.join(TB_LOGS_DIR, experiment, mode, run_name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def log_metrics(metrics:dict, epoch:int, step:int, global_step:int,
                metrics_computer=BinClassificationMetrics,
                tb_writer:SummaryWriter=None):
    metrics_computer.log_metrics(metrics, epoch=epoch, step=step)
    # Tensorboard
    if tb_writer:
        for metric_name, value in metrics.items():
            if metric_name == 'loss':
                tb_writer.add_scalar('Loss/CrossEntropy', value, 
                                     global_step)
            else:
                tb_writer.add_scalar('Metrics/' + metric_name, 
                                     value, global_step)

def train_step(
        model:Model,
        batch:tuple,
        optimizer,
        loss,
        metrics_computer:BinClassificationMetrics,
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


@status_handler
def main(train_data:str,
         test_data:str,
         config:Union[str, dict]='config.yaml',
         epochs:int=15,
         batch_size:int=500,
         experiment:str='experiment',
         no_save:bool=False,
         use_manager:bool=True,
         tensorboard:bool=False,
         data_shuffle:bool=True,
         log_step:int=1,
         comment:str=None,
    ):
    """
    train_data(str): path/to/train/data
    test_data(str): path/to/test/data
    config (str, dict): config dict or path/to/config.yaml
    experiment (str): experiment name
    use_manager (bool): whether to manage experiment
    tensorboard (bool): whether to log step metrics to TB
    data_shuffle (bool): whether to shuffle data
    log_step (int): interval of loggoing step metrics
    comment (str): postfix for experiment run name
    """
    experiment = experiment.lower().replace(' ', '_')
    global manager

    # Load config
    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config_yaml)
    train_params = config["train"]
    test_params = config["test"]
    data_params = config["data"]
    manager_params = config["manager"]
    train_logger, test_logger = None, None

    # Load train data
    train_set = MyDataset(train_data, params=data_params)
    train_data_size = len(train_set)
    sampler = BucketingSampler(train_set, batch_size, shuffle=data_shuffle)
    train_set = CudaDataLoader(train_set, collate_fn=train_set.collate, 
                               pin_memory=True, num_workers=4,
                               batch_sampler=sampler)
    train_steps = len(train_set)  # number of train batches

    # Load test data
    test_set = MyDataset(test_data, params=data_params)
    test_data_size = len(test_set)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(test_set, collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)
    test_steps = len(test_set)  # number of test batches
    
    # Define metadata
    metadata = {
            "train_data": train_data,
            "test_data": test_data,
            "batch_size": batch_size,
            "train_data_size": train_data_size,
            "train_steps": train_steps,
            "test_data_size": test_data_size,
            "test_steps": test_steps,
    }
    utils.pprint_dict(metadata)

    # Create experiment
    if not no_save:
        runs_dir = os.path.join(EXPERIMENTS_DIR,
                                experiment,
                                'train')
        if not osp.exists(runs_dir):
            os.makedirs(runs_dir)
        run_name = '{:03d}'.format(get_new_run_id(runs_dir))
        if comment:
            run_name += '_' + comment
        # init dirs
        run_dir = osp.join(runs_dir, run_name)
        os.makedirs(run_dir)
        os.makedirs(osp.join(run_dir, 'weights/'))
        # save config to run_dir
        utils.dict2yaml(config, osp.join(run_dir, 'config.yaml'))
        utils.dict2yaml(metadata, osp.join(run_dir, 'meta.yaml'))
        # init files for log metrics (in csv format)
        train_logfile = osp.join(run_dir, 'train.csv')
        train_logger = utils.get_logger('train', train_logfile)
        test_logfile = osp.join(run_dir, 'test.csv')
        test_logger = utils.get_logger('test', test_logfile)
        # set path to best weights.pt
        best_weights_path = osp.join(run_dir, f"weights/best.pt")
        print(f"Experiment storage: '{run_dir}'")

        # Init manager
        if use_manager:
            tags = {
                'mode': 'train',
                'epochs': str(epochs),
            }
            manager = MLFlowManager(
                url=manager_params["url"],
                experiment=experiment,
                run_name='train-' + run_name,
                tags=tags
            )
            manager.log_hyperparams(manager_params["hparams"])
            manager.log_config(config)
            manager.log_config(metadata, 'meta.yaml')
            print(f"Manager experiment run name: {'train-' + run_name}")

    # Define model
    model_name = train_params["model"]
    model_params = config["model"][model_name]
    model_params["weights"] = train_params["pretrained"] 
    # None of path/to/model.pt
    model = model_init(model_name,
                       model_params,
                       train=True,
                       device="cuda:0")

    # Define optimizer
    opt = train_params["opt"]
    if opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_params["learning_rate"], 
            weight_decay=train_params['weight_decay'])
    else:
        raise Exception(f"No optimizer: '{opt}'")

    # Tensorboard writer
    if not no_save and tensorboard:
        writer_train = create_tb_writer(experiment, run_name, 'train')
        writer_test = create_tb_writer(experiment, run_name, 'test')
    else:
        writer_test = writer_train = None

    # Define loss
    loss_name = train_params["loss"]
    loss_params = config["loss"][loss_name]
    loss = init_loss(loss_name, loss_params, device="cuda:0")

    # Init train metrics computer
    compute_metrics = train_params["metrics"]
    train_metrics_computer = BinClassificationMetrics(
                               step=True, epoch=True,
                               compute_metrics=compute_metrics,
                               logger=train_logger)

    # Init test metrics computer
    compute_metrics = test_params["metrics"]
    test_metrics_computer = BinClassificationMetrics(
                                step=True, epoch=True,
                                compute_metrics=compute_metrics,
                                logger=test_logger)

    best_metrics = {
        'TP': 0,
        'FN': 1,
        'FN': 1,
        'TN': 0,
    }
    train_iter = test_iter = 0
    for ep in range(1, epochs + 1):
        print(f"\n{ep}/{epochs} Epoch...")
        if manager:
            manager.add_tags({'current_epoch': ep})

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
            if (i+1) % log_step == 0:
                log_metrics(metrics, epoch=ep, step=i+1, 
                            global_step=train_iter,
                            metrics_computer=train_metrics_computer,
                            tb_writer=writer_train)
            train_iter += 1
                
        # Saving
        if not no_save:
            weights_path = osp.join(run_dir, f"weights/{ep}.pt")
            model.save(weights_path)
            print(f"Weights save: '{weights_path}'")

        print('------------- Test ---------------')

        for i, batch in enumerate(test_set):
            metrics = test_step(
                model=model,
                batch=batch,
                loss=loss,
                metrics_computer=test_metrics_computer
            )
            log_metrics(metrics, epoch=ep, step=i+1, 
                        global_step=test_iter,
                        metrics_computer=test_metrics_computer,
                        tb_writer=writer_test)
            test_iter += 1
            
        print("Average metrics:")
        metrics = test_metrics_computer.summary()
        for k, v in metrics.items():
            print(f"{k}: {v}")

        if manager:
            manager.log_step_metrics(metrics, step=ep)

        # Сheck whether it's the best metrics
        if best_metrics is None or \
                get_better_metrics(metrics, best_metrics) is metrics:
            best_metrics = metrics
            print('New best results')
            # save best weights
            if not no_save:
                model.save(best_weights_path)
                print(f"Weights save: '{best_weights_path}'")

        test_metrics_computer.reset_summary()

        # ---- END OF EPOCH

    print("Best metrics:")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")
    if manager:
        manager.log_summary_metrics(best_metrics)
        config["model"][model_name]["weights"] = best_weights_path
        utils.dict2yaml(config, osp.join(run_dir, 'config.yaml'))
        manager.log_config(osp.join(run_dir, 'config.yaml'))

    if not no_save:
        if manager:
            manager.set_status("FINISHED")
        if tensorboard:
            writer_train.close()
            writer_test.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--config', '-cfg', type=str, 
                        default='config.yaml', 
                        help='path/to/config.yaml')
    parser.add_argument('--train-data', type=str, 
                        default='data/train_manifest.csv')
    parser.add_argument('--test-data', type=str, 
                        default='data/test_manifest.csv')
    parser.add_argument('--batch-size', '-bs', type=int, 
                        default=100)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--no-save', '-ns', action='store_true', 
                        default=False, 
                        help='no save results')
    parser.add_argument('--experiment', '-exp', type=None, 
                        default='experiment', 
                        help='Name of existed MLFlow experiment')
    parser.add_argument('--manager', '-mng', action='store_true', 
                        dest='use_manager', default=False, 
                        help='whether to use ML experiment manager')
    parser.add_argument('--tensorboard', '-tb', action='store_true', 
                        default=False, 
                        help='whether to use Tensorboard')
    parser.add_argument('--comment', '-m', type=str, default=None, 
                        help='Postfix for experiment run name')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    main(**args)