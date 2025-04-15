from typing import Union, Tuple
import os
import os.path as osp
import numpy as np
from math import isnan
import argparse
from glob import glob
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from .data import CudaDataLoader, BucketingSampler, NoiseClassificationDataset
from .eval import test_step
from .metrics import init_loss, ClassificationMetrics, Loss
from .models import init_model, BaseModel
from . import utils
from .utils.manager import ClearMLManager
from .utils import EXPERIMENTS_DIR
try:
    import optuna
except ImportError:
    pass


def get_run_name(runs_dir:str, run_id:int) -> str:
    if not osp.exists(runs_dir):
        raise ValueError(f"Experiment dir doesn't exist: '{runs_dir}'")
    tmpt = osp.join(runs_dir, '{:03d}*'.format(run_id))
    runs = glob(tmpt)
    # ['path/to/experiment/002_comment/']
    if not runs:
        raise ValueError(f"Train run '{tmpt}' doesn't exist")
    if len(runs) > 1:
        raise ValueError(f"There multiple train runs with ID {run_id}: "\
                         f"{runs}")
    run_name = osp.split(runs[0])[1]
    return run_name


def get_last_epoch_weights(weights_dir:str) -> Union[str, None]:
    tmpt = osp.join(weights_dir, "[0-9]*.pt")
    weights = glob(tmpt)  # ['path/to/2.pt', 'path/to/12.pt']
    if not weights:
        return
    last_epoch = max([int(osp.splitext(osp.split(w)[1])[0]) \
                      for w in weights])
    return last_epoch


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
    score = - metrics["CrossEntropyLoss"]
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

def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    # (B, )
    lam = torch.from_numpy(lambd).to("cuda:0")
    # lam = torch.FloatTensor(lambd, device="cuda:0")
    return rn_indices, lam


def train_step(
        model:BaseModel,
        batch:Tuple[Tensor, Tensor],
        optimizer,
        loss_computer:Loss,
        metrics_computer:ClassificationMetrics,
        train_params:dict,
        mixup_alpha:float=0
        ) -> dict:
    # clean previous grads
    optimizer.zero_grad()          

    x, target = batch
    # target - one hot (B, C) ?

    if mixup_alpha > 0:
        x = model.before_mixup(x)

        bs = x.shape[0]
        rn_indices, lam = mixup(bs, mixup_alpha)
        x = x * lam.reshape(bs, 1, 1, 1) + \
            x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
        logits, probs = model.after_mixup(x)

        loss, loss_values = loss_computer(logits, target, rn_indices, lam)

    else:
        logits, probs = model(x)
        # logits - before activation (for loss)
        # probs - after activation   (for acc)

        loss, loss_values = loss_computer(logits, target)

    # Check if loss is nan
    if torch.isnan(loss) or \
            any([isnan(v) for v in loss_values.values()]):
        message = f"Loss is NaN"
        raise Exception(message)

    # backpropogation
    # just calculating weight.grad
    # weights are not updated here!
    loss.backward()

    # prevent exploding gradient
    clip_grad_norm_(model.parameters(), train_params['grad_norm'])

    # w_new = w_old - lr * w.grad
    optimizer.step()

    # check if grads are not NaN
    model.validate_grads()

    # metrics computing
    metrics = metrics_computer.step_metrics(probs, target,
                                            add_summary=True,
                                            precomputed=loss_values)
    return metrics


def train(train_data:str,
         test_data:str,
         config:Union[str, dict]='config.yaml',
         epochs:int=15,
         batch_size:int=500,
         cache_size:int=1000,
         gpu_id:int=0,
         experiment:str='experiment',
         resume:int=None,
         no_save:bool=False,
         clearml:bool=False,
         data_shuffle:bool=True,
         log_step:int=1,
         comment:str=None,
         task_name:str="train",
         trial=None
    ):
    """
    train_data(str): path/to/train/data
    test_data(str): path/to/test/data
    config (str, dict): config dict or path/to/config.yaml
    experiment (str): experiment name
    resume (int): train experiment run to continue training
        from last epoch. 
        Config and last weights will be loaded from this experiment.
        It's also able to define new data
    clearml (bool): whether to manage experiment with ClearML
    data_shuffle (bool): whether to shuffle data
    cache_size (int): how much audio samples to store in RAM 
        for faster batch generation
    log_step (int): interval of loggoing step metrics
    comment (str): postfix for experiment run name
    task_name (str): name of runs subdir under the experiment
    trial (optuna.Trial, None): for early stopping 
        when using hypertuning with Optuna
    """
    experiment = experiment.lower().replace(' ', '_')
    train_logger, test_logger, loss = None, None, None
    manager, run_dir, run_name = None, None, None
    summary_file = None
    weights = None  # pretrained weights or from previous run
    start_epoch = 1
    hparams = dict()

    # check optuna is available
    if trial is not None:
        try:
            from optuna import TrialPruned
        except:
            raise ValueError("For given trial early stopping with "\
                             "Optuna must be available. Check "\
                             "'from optuna import TrialPruned'")

    # Validate device
    num_valid_gpus = torch.cuda.device_count()
    if gpu_id >= num_valid_gpus:
        raise ValueError(f"Only {num_valid_gpus} GPUs are available")
    device = f"cuda:{gpu_id}"

    runs_dir = os.path.join(EXPERIMENTS_DIR,
                            experiment,
                            task_name)
    if resume:
        # get train run dir
        run_name = get_run_name(runs_dir, resume)
        run_dir = osp.join(runs_dir, run_name)
        print(f"Resume training from '{run_dir}'")
        # load config from this train run
        config = osp.join(run_dir, "config.yaml")
        # load last epoch weights
        last_epoch = get_last_epoch_weights(run_dir + '/weights')  # {ep}.pt
        if last_epoch:
            start_epoch = last_epoch + 1
            weights = osp.join(run_dir, f'weights/{last_epoch}.pt')

    # Load config
    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config_yaml)
    else:
        config_yaml = '/tmp/config.yaml'
        utils.dict2yaml(config, config_yaml)

    # Create experiment
    if not no_save:
        os.makedirs(runs_dir, exist_ok=True)
        if not run_name:
            new_run_num = utils.get_next_exprun(runs_dir, "\d{3}", return_num=True)
            run_name = '{:03d}'.format(new_run_num)
            if comment:
                run_name += '_' + comment

        # init dirs
        run_dir = osp.join(runs_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(osp.join(run_dir, 'weights/'), exist_ok=True)
        
        # Init manager
        if clearml:
            params = config["manager"]["clearml"]
            params.update({
                "experiment": experiment,
                "run_name": task_name + '-' + run_name,
                "train": True,
                "resume": resume
            })
            manager = ClearMLManager(**params)
            manager.set_iterations(epochs)
            # log and update config if it's defined in experiment
            config_yaml = manager.log_config(config_yaml)
            config = utils.config_from_yaml(config_yaml)
            # log and update hparams if it was changed
            # Hyperparams
            hparams = config["manager"]["hparams"]
            hparams = manager.log_hyperparams(hparams)  
            # hparams can be update here using ClearML
            utils.overwrite_hparams(config, hparams)
            print("Hyperparams:")
            utils.pprint_dict(hparams)
            print(f"Manager experiment run name: {task_name + '-' + run_name}")
            
        # save final config
        config_yaml = osp.join(run_dir, 'config.yaml')
        utils.dict2yaml(config, config_yaml)
        # init files for log metrics (in csv format)
        train_logfile = osp.join(run_dir, 'train.csv')
        train_logger = utils.get_logger('train', train_logfile)
        test_logfile = osp.join(run_dir, 'test.csv')
        test_logger = utils.get_logger('test', test_logfile)
        # set path to best weights.pt
        best_weights_path = osp.join(run_dir, f"weights/best.pt")
        summary_file = osp.join(run_dir, 'summary.txt')
        print(f"Experiment storage: '{run_dir}'")

    # Config is final here
    train_params = config["train"]
    test_params = config["eval"]
    classes = utils.read_classes(config["classes"])
    sr = config["sr"]
    normalize = config["preprocess"]["normalize"]
    mixup_alpha = train_params["mixup"]

    # Load train data
    preprocess_cfg = config["preprocess"]  # outside model
    train_set = NoiseClassificationDataset(train_data, classes=classes,
                                 sr=sr, normalize=normalize,
                                 cache_size=cache_size,
                                 preprocess_cfg=preprocess_cfg)
    train_data_size = len(train_set)
    sampler = BucketingSampler(train_set, batch_size,
                               shuffle=data_shuffle)
    train_set = CudaDataLoader(gpu_id, train_set, 
                               collate_fn=train_set.collate, 
                               pin_memory=True, num_workers=8,
                               batch_sampler=sampler)
    train_steps = len(train_set)  # number of train batches

    # Define model
    model_cfg = config["model"]
    weights = weights or train_params["pretrained"]
    print(f'Start training with weights: {weights}')
    # None or path/to/model.pt
    model = init_model(n_classes=len(classes),
                       model_cfg=model_cfg,
                       weights=weights,
                       training=True,
                       device=device)

    # Load test data
    test_set = NoiseClassificationDataset(test_data, classes=classes,
                                sr=sr, normalize=normalize,
                                cache_size=cache_size,
                                preprocess_cfg=preprocess_cfg)
    test_data_size = len(test_set)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(gpu_id, test_set, 
                              collate_fn=test_set.collate,
                              pin_memory=True, num_workers=8,
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
        
    # Init train metrics computer
    train_metrics_computer = ClassificationMetrics(
                                classes=classes,
                                step_metrics=train_params["step_metrics"],
                                summary_metrics=train_params["sum_metrics"],
                                step=True, epoch=True,
                                logger=train_logger,
                                log_title=not resume)

    # Init test metrics computer
    test_metrics_computer = ClassificationMetrics(
                                classes=classes,
                                step_metrics=test_params["step_metrics"],
                                summary_metrics=test_params["sum_metrics"],
                                step=True, epoch=True,
                                logger=test_logger,
                                log_title=not resume)
    
    # Define loss
    loss_cfg = config["loss"]
    loss = init_loss(loss_cfg, device=device)

    # Define optimizer
    assert not hparams or train_params["learning_rate"] == hparams["train"]["learning_rate"]
    opt = train_params["opt"]
    if opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_params["learning_rate"], 
            weight_decay=train_params['weight_decay']
        )
    elif opt == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params['weight_decay'],
            nesterov=train_params["nesterov"],
            momentum=train_params["momentum"]
        )
    else:
        raise Exception(f"No optimizer: '{opt}'")
    
    # Learning rate scheduler
    lr_scheduler_cfg = train_params["lr_scheduler"]
    if lr_scheduler_cfg["use"]:
        sch_lambda = \
            utils.exp_warmup_linear_down(
                warmup=lr_scheduler_cfg["warmup"],
                rampdown_length= lr_scheduler_cfg["rd_len"],
                start_rampdown=lr_scheduler_cfg["rd_start"],
                last_value=lr_scheduler_cfg["last_lr"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sch_lambda)
    else:
        lr_scheduler = None
        
    best_metrics, best_epoch = dict(), None
    if manager:
        manager.set_iterations(epochs - start_epoch + 1)
    
    for ep in range(start_epoch, epochs + 1):
        print(f"\n{ep}/{epochs} Epoch...")
        model.train()
        if manager:
            # like progress bar
            manager.log_iteration(ep - start_epoch)

        train_set.shuffle(ep)
        # Progress bary
        if not no_save:
            train_batches = utils.get_progress_bar(train_set, 
                                               total=train_steps, 
                                               title=f"Epoch {ep}")
            # doesn't always work with enumerate
        else:
            train_batches = train_set

        # Train epoch is starting ...
        i = 0
        for batch in train_batches:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_computer=loss,
                metrics_computer=train_metrics_computer,
                train_params=train_params,
                mixup_alpha=mixup_alpha,
            )
            # filter out not required metrics
            if (i+1) % log_step == 0:
                train_metrics_computer.log_metrics(
                    metrics, epoch=ep, step=i+1)
            i += 1
        # Train of epoch ends
            
        if lr_scheduler:
            lr_scheduler.step()
            
        # Saving
        if not no_save:
            weights_path = osp.join(run_dir, f"weights/{ep}.pt")
            model.save(weights_path)
            print(f"Weights save: '{weights_path}'")

        print('------------- Test ---------------')
        model.eval()
        test_set.shuffle(ep)
        # Progress bar
        if not no_save:
            test_batches = utils.get_progress_bar(test_set, 
                                              total=test_steps, 
                                              title=f"Epoch {ep}")
        else:
            test_batches = test_set

        i = 0
        for batch in test_batches:
            metrics = test_step(
                model=model,
                batch=batch,
                loss_computer=loss,
                metrics_computer=test_metrics_computer,
            )
            test_metrics_computer.log_metrics(
                    metrics, epoch=ep, step=i+1)
            i += 1
            
        # Summary metrics after this epoch
        print(f"\n--- Train summary after epoch {ep}---")
        train_summary = train_metrics_computer.get_summary()
        train_metrics_computer.pprint(train_summary, line=False)
        
        print(f"\n--- Test summary after epoch {ep}---")
        test_summary = test_metrics_computer.get_summary()
        test_metrics_computer.pprint(test_summary, line=False, 
                                     with_conf_matrix=True,
                                     duplicate_file=summary_file)

        # Сheck whether it's the best metrics
        if not best_metrics or \
                get_better_metrics(test_summary, best_metrics) is test_summary:
            best_metrics = dict(test_summary)  # copy
            best_epoch = ep
            print('New best results')
            # save best weights
            if not no_save:
                model.save(best_weights_path)
                print(f"Weights save: '{best_weights_path}'")

        # Save test metrics after current epoch
        if manager:
            conf_matrix = test_summary.pop("conf_matrix", None)
            manager.log_step_metrics(train_summary, step=ep - start_epoch, prefix="train")
            manager.log_step_metrics(test_summary, step=ep - start_epoch, prefix="test")

        # early stopping when using hypertuning with Optuna
        if trial is not None:
            objective = config["hypertune"]["objective"]
            if len(objective) > 1:
                raise NotImplementedError("Multiobjective optimization is not implemented yet")
            objective = objective[0]
            if "train_" in objective:
                obj_value = utils.get_obj_value(train_summary, objective)
            else:
                obj_value = utils.get_obj_value(test_summary, objective)
            trial.report(obj_value, step=ep-1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        test_metrics_computer.reset_summary()
        train_metrics_computer.reset_summary()

        # ---- END OF EPOCH


    # BEST result in this experiment
    print("\n--- Best metrics ---")
    print(f"Best epoch: {best_epoch}")
    test_metrics_computer.pprint(best_metrics, line=False,
                                 with_conf_matrix=True,
                                 duplicate_file=summary_file)
    
    if not no_save:
        # Save BEST
        model_cfg["weights"] = best_weights_path
        utils.dict2yaml(config, config_yaml)
        if manager:
            conf_matrix = best_metrics.pop("conf_matrix", None)
            if conf_matrix is not None:
                manager.log_confusion_matrix(conf_matrix, 
                                             classes=classes,
                                             normalize=test_params["norm_conf_matrix"])
            manager.log_summary_metrics(best_metrics)
            manager.log_config(config_yaml)
            manager.add_tags([f"best: {best_epoch}.pt"])
            manager.close()

    return best_metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, 
                        default='config.yaml', 
                        help='path/to/config.yaml')
    parser.add_argument('--train-data', type=str, 
                        default='/app/data/esc50-5s-train.csv')
    parser.add_argument('--test-data', type=str, 
                        default='/app/data/esc50-5s-test.csv')
    parser.add_argument('--batch-size', '-bs', type=int, 
                        default=50)
    parser.add_argument('--gpu', type=int, dest="gpu_id", default=0,
                        help='which GPU to use')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--no-save', action='store_true', 
                        default=False, 
                        help='no save results')
    parser.add_argument('--experiment', '-exp', type=str, 
                        default='experiment', 
                        help='Name of experiment')
    parser.add_argument('--clearml', action='store_true', 
                        default=False, 
                        help='whether to use ClearML for experiment manager')
    parser.add_argument('--cache-size', '-cs', type=int, 
                        default=1000,
                        help="how much audio samples to store in RAM"\
                             "for faster batch generation")
    parser.add_argument('--resume', type=int, 
                        default=None, 
                        help='Train experiment run to resume training')
    parser.add_argument('--comment', '-m', type=str, default=None, 
                        help='Postfix for experiment run name')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    parser.add_argument('--task-name', type=str, default="train",
                        help='Name of runs subdir under the experiment')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    train(**args)
