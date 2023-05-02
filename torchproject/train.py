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
from math import isnan
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from .data import CudaDataLoader, BucketingSampler, AudioDataset
from . import utils
from .metrics import init_loss, BinClassificationMetrics, Loss
from .models import Model, model_init
from .utils.manager import MLFlowManager, ClearMLManager
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

def get_metrics_score(metrics:dict) -> float:
    """ Score of metrics for best criteria 
    For example: score = 2 * Recall + Precision
    NOTE: the higher score is better
    """
    score = 2 * metrics["Recall"] + metrics["Precision"]
    return score

def get_better_metrics(metrics1:dict, metrics2:dict) -> dict:
    """
    Here you need define best criteria
    Returns:
        dict: one of two metrics dict which is better
    """
    if get_metrics_score(metrics1) > get_metrics_score(metrics2):
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
        loss_computer:Loss,
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

    loss, loss_values = loss_computer(logits, target.float())

    # Check if loss is nan
    if torch.isnan(loss) or \
            any([isnan(v) for v in loss_values.values()]):
        message = f"Loss is NaN"
        raise Exception(message)

    # обратное распр-е ошибки.
    # для каждого параметра модели w считает w.grad
    # здесь НЕ обновляются веса!
    loss.backward()

    clip_grad_norm_(model.parameters(), train_params['grad_norm'])
    # prevent exploding gradient

    # здесь обновление весов
    # w_new = w_old - lr * w.grad
    optimizer.step()

    # check if grads are not NaN
    model.validate_grads()

    # metrics computing
    metrics = metrics_computer.compute(probs, target,
                                       accumulate=False)
    metrics.update(loss_values)
    return metrics


@status_handler
def main(train_data:str,
         test_data:str,
         config:Union[str, dict]='config.yaml',
         epochs:int=15,
         batch_size:int=500,
         gpu_id:int=0,
         experiment:str='experiment',
         no_save:bool=False,
         use_mlflow:bool=False,
         use_clearml:bool=False,
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
    train_logger, test_logger = None, None

    # Validate device
    num_valid_gpus = torch.cuda.device_count()
    if gpu_id >= num_valid_gpus:
        raise ValueError(f"Only {num_valid_gpus} GPUs are available")
    device = f"cuda:{gpu_id}"

    # Load config
    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config_yaml)
    else:
        config_yaml = '/tmp/config.yaml'
        utils.dict2yaml(config, config_yaml)
    manager_params = config["manager"]

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
        
        # Init manager
        if use_mlflow or use_clearml:
            params = {
                "experiment": experiment,
                "run_name": 'train-' + run_name,
                "train": True,
            }
            if use_clearml:
                params.update(manager_params["clearml"])
                manager = ClearMLManager(**params)
            else:
                params.update(manager_params["mlflow"])
                manager = MLFlowManager(**params)

            manager.set_iterations(epochs)
            # log and update config if it's defined in experiment
            config_yaml = manager.log_config(config_yaml)
            config = utils.config_from_yaml(config_yaml)
            hparams = config["manager"]["hparams"]
            # log and update if it was changed
            hparams = manager.log_hyperparams(hparams)  
            print(f"Manager experiment run name: {'train-' + run_name}")
        
        # save config to run_dir
        config_yaml = osp.join(run_dir, 'config.yaml')
        utils.dict2yaml(config, config_yaml)
        # init files for log metrics (in csv format)
        train_logfile = osp.join(run_dir, 'train.csv')
        train_logger = utils.get_logger('train', train_logfile)
        test_logfile = osp.join(run_dir, 'test.csv')
        test_logger = utils.get_logger('test', test_logfile)
        # set path to best weights.pt
        best_weights_path = osp.join(run_dir, f"weights/best.pt")
        print(f"Experiment storage: '{run_dir}'")

    # Update config with hparams
    utils.update_given_keys(config, hparams)
    train_params = config["train"]
    test_params = config["test"]
    preprocess_params = config["preprocess"]
    n_classes = config["n_classes"]

    # Load train data
    train_set = AudioDataset(train_data, n_classes=n_classes, 
                             preprocess_params=preprocess_params)
    train_data_size = len(train_set)
    sampler = BucketingSampler(train_set, batch_size, shuffle=data_shuffle)
    train_set = CudaDataLoader(gpu_id, train_set, 
                               collate_fn=train_set.collate, 
                               pin_memory=True, num_workers=4,
                               batch_sampler=sampler)
    train_steps = len(train_set)  # number of train batches

    # Load test data
    test_set = AudioDataset(test_data, n_classes=n_classes,
                            preprocess_params=preprocess_params)
    test_data_size = len(test_set)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(gpu_id, test_set, 
                              collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)
    test_steps = len(test_set)  # number of test batches
    
    # Define metadata
    metadata = {
            "storage": run_dir,
            "train_data": train_data,
            "test_data": test_data,
            "batch_size": batch_size,
            "train_data_size": train_data_size,
            "train_steps": train_steps,
            "test_data_size": test_data_size,
            "test_steps": test_steps,
            "log_step": log_step
    }
    utils.pprint_dict(metadata)
    if not no_save:
        meta_yaml = osp.join(run_dir, 'meta.yaml')
        utils.dict2yaml(metadata, meta_yaml)
        if manager:
            manager.log_metadata(metadata)

    # Define model
    model_name = config["model"]
    model_params = config["models"][model_name]
    model_params["weights"] = train_params["pretrained"] 
    # None of path/to/model.pt
    model = model_init(model_name,
                       model_params,
                       train=True,
                       device=device)

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
    loss = init_loss(loss_name, loss_params, device=device)

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

    best_metrics = None
    train_iter = test_iter = 0
    for ep in range(1, epochs + 1):
        print(f"\n{ep}/{epochs} Epoch...")
        if manager:
            manager.log_iteration(ep)

        model.train()
        train_set.shuffle(ep)
        for i, batch in enumerate(train_set):
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_computer=loss,
                metrics_computer=train_metrics_computer,
                train_params=train_params,
            )
            # filter out not required metrics
            metrics = {k: metrics[k] for k in train_params["metrics"]}
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
                loss_computer=loss,
                metrics_computer=test_metrics_computer
            )
            # filter out not required metrics
            metrics = {k: metrics[k] for k in test_params["metrics"]}
            log_metrics(metrics, epoch=ep, step=i+1, 
                        global_step=test_iter,
                        metrics_computer=test_metrics_computer,
                        tb_writer=writer_test)
            test_iter += 1
            
        print("Average metrics:")
        metrics = test_metrics_computer.summary()
        metrics = {k: metrics[k] for k in test_params["metrics"]}
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
    if not no_save:
        model_params["weights"] = best_weights_path
        utils.dict2yaml(config, config_yaml)
        if manager:
            manager.log_summary_metrics(best_metrics)
            manager.log_config(config_yaml)
            manager.set_status("FINISHED")
            manager.close()
        if tensorboard:
            writer_train.close()
            writer_test.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--config', '-cfg', type=str, 
                        default='config.yaml', 
                        help='path/to/config.yaml')
    parser.add_argument('--train-data', type=str, 
                        default='data/processed/train_manifest.v1.csv')
    parser.add_argument('--test-data', type=str, 
                        default='data/processed/test_manifest.v1.csv')
    parser.add_argument('--batch-size', '-bs', type=int, 
                        default=100)
    parser.add_argument('--gpu', type=int, dest="gpu_id", default=0,
                        help='which GPU to use')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--no-save', '-ns', action='store_true', 
                        default=False, 
                        help='no save results')
    parser.add_argument('--experiment', '-exp', type=None, 
                        default='experiment', 
                        help='Name of existed MLFlow experiment')
    parser.add_argument('--mlflow', action='store_true', 
                        dest='use_mlflow', default=False, 
                        help='whether to use MLFlow for experiment manager')
    parser.add_argument('--clearml', action='store_true', 
                        dest='use_clearml', default=False, 
                        help='whether to use ClearML for experiment manager')
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
