import torch
import os
import os.path as osp
from typing import Union, Tuple
import argparse
from math import isnan
from torch import Tensor
from .models import MLModel
from .data import CudaDataLoader, BucketingSampler, BlockChainDataset
from . import utils
from .utils.manager import ClearMLManager
from .models import model_init, MLModel, IterativeModel
from .metrics import init_loss, ClassificationMetrics, Loss
from .utils import EXPERIMENTS_DIR


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
    

def test_step(
        model:MLModel,
        batch:Tuple[Tensor, Tensor],
        metrics_computer:ClassificationMetrics,
        loss_computer:Loss=None,
        ) -> dict:
    x, target = batch
    with torch.no_grad():
        logits, probs = model.predict(x)
    # logits - before activation (for loss)
    # probs - after activation   (for acc)
        
    # Metrics computing
    metrics = metrics_computer.compute(probs.cpu(), target.cpu(),
                                       accumulate=True)
    # CrossEntropy loss
    if loss_computer:
        loss, loss_values = loss_computer(logits, target)
        # Check if loss is nan
        if torch.isnan(loss) or \
            any([isnan(v) for v in loss_values.values()]):
            message = f"Loss is NaN"
            raise Exception(message)
        metrics.update(loss_values)
    
    return metrics


def test(data:str,
         config:Union[str, dict]='config.yaml',
         batch_size:int=500,
         gpu_id:int=0,
         no_save:bool=False,
         experiment:str='experiment',
         run_id:int=None,
         weights:str='best.pt',
         clearml:bool=False,
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
    logger, run_dir, manager = None, None, None
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
        if clearml:
            params = config["manager"]
            params.update({
                "experiment": experiment,
                "run_name": 'test-' + run_name,
                "train": False,
            })
            manager = ClearMLManager(**params)
            # log and update config if it's defined in experiment
            config_yaml = manager.log_config(config_yaml)
            config = utils.config_from_yaml(config_yaml)
            # log and update hparams if it was changed
            hparams = config["manager"]["hparams"]
            hparams = manager.log_hyperparams(hparams)  
            print(f"Manager experiment run name: {'train-' + run_name}")

        # save config
        config_yaml = osp.join(run_dir, 'config.yaml')
        utils.dict2yaml(config, config_yaml)
        # init files for log metrics
        logfile = osp.join(run_dir, 'test.csv')
        logger = utils.get_logger('test', logfile)
        print(f"Experiment storage: '{run_dir}'")

    # Hyperparams overwrite config params
    utils.update_given_keys(config, hparams)
    params = config["test"]
    classes = config["classes"]
    model_cfg = config["model"]
    # Redefine weights
    if weights:
        model_cfg["weights"] = weights

    # Load test data
    test_set = BlockChainDataset(data, classes=classes)
    data_size = len(data_size)
    sampler = BucketingSampler(test_set, batch_size, shuffle=data_shuffle)
    test_set = CudaDataLoader(gpu_id, test_set, 
                              collate_fn=test_set.collate,
                              pin_memory=True, num_workers=4,
                              batch_sampler=sampler)
    test_steps = len(test_set)  # number of test batches

    # Add specific info
    if manager:
        manager.set_iterations(test_steps)
        weights_name = osp.split(model_cfg["weights"])[1]
        manager.add_tags([f"weights: {weights_name}"])

    # Define metadata
    metadata = {
            "data": data,
            "batch_size": batch_size,
            "data_size": data_size,
            "test_steps": test_steps,
            "storage": run_dir,
            "weights": model_cfg["weights"],
            
    }
    utils.pprint_dict(metadata)
    if not no_save:
        meta_yaml = osp.join(run_dir, 'meta.yaml')
        utils.dict2yaml(metadata, meta_yaml)
        if manager:
            manager.log_metadata(metadata)

    # Define model
    if not model_cfg["weights"]:
        raise ValueError("Weights are not defined")
    model = model_init(model_cfg,
                       train=False,
                       device=device)
    
    iterative_train = isinstance(model, IterativeModel)
    if iterative_train:
        # Define loss
        loss_name = config["loss"]
        loss_params = config["loss"][loss_name]
        loss = init_loss(loss_name, loss_params, device=device)

    # Init test metrics computer
    metrics_computer = ClassificationMetrics(
                                classes=classes,
                                metrics=params["step_metrics"],
                                step=True, epoch=True,
                                logger=logger)

    test_set.shuffle(15)
    for step, batch in enumerate(test_set):
        metrics = test_step(
            model,
            batch,
            loss,
            metrics_computer
        )
        if (step + 1) % log_step == 0:
            metrics_computer.log_metrics(metrics, step=step)
            if manager:
                manager.log_step_metrics(metrics, step)

    sum_metrics = metrics_computer.summary(params["sum_metrics"])
    print("\n--- Summary metrics ---")
    metrics_computer.pprint(sum_metrics, line=False)
    # Print summary conf matrix
    conf_matrix = metrics_computer.sum_conf_matrix
    metrics_computer.pprint_conf_matrix(conf_matrix)
    
    if manager:
        manager.log_summary_metrics(sum_metrics)
        if params["plot_conf_matrix"]:
            manager.log_confusion_matrix(conf_matrix)
        if params["plot_roc"]:
            raise NotImplementedError()
        if params["plot_pr"]:
            raise NotImplementedError()
        manager.close()



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
    parser.add_argument('--clearml', action='store_true', 
                        dest='use_clearml', default=False, 
                        help='whether to use ClearML for experiment manager')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    parser.add_argument('--comment', '-m', type=str, default=None, 
                    help='Postfix for experiment run name')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    test(**args)
