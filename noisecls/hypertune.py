import os.path as osp
from typing import Union, List
import json
from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
try:
    import optuna
    from optuna.samplers import BaseSampler
except ImportError:
    print("Optuna package is not supported. Can not be used.")
    pass
from .train import train
from .utils import EXPERIMENTS_DIR
from . import utils

"""
Job is one experiment

hparams.json
[
    {
        "name": "Args/batch_size",
        "values": [5, 10, 12]
    },
    {
        "name": "hparams/train/learning_rate",
        "uniform": [0.01, 0.1, 0.03]
    },
    {
        "name": "General/param",
        "uniform": [0, 100, 20]
    }
]
"""
HPARAMS_JSON = "/app/noisecls/utils/hparams.json"

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


def parse_clearml_hparams(json_path) -> list:
    """
    Parse ClearML hparams for tuning from JSON
    Format like
    [
        {
            "name": "Args/batch_size",
            "values": [5, 10, 12]
        },
        {
            "name": "hparams/train/learning_rate",
            "uniform": [0.01, 0.1, 0.03]
        },
        {
            "name": "General/param",
            "uniform": [0, 100, 20]
        }
    ]
    Returns:
        List[Parameter]: list of clearml.automation.parameters.Parameter
    """
    with open(json_path, 'rb') as f:
        data = json.load(f)  # list[dict]
    hparams = []
    for d in data:
        d : dict
        name = d["name"]
        values = d.get("values")
        uniform = d.get("uniform")
        if not utils.xor(values, uniform):
            raise ValueError("Either 'values' or 'uniform' must be defined")
        
        if values:
            hparam = DiscreteParameterRange(name, values=values)
        else:
            if len(uniform) != 3:
                msg = "Uniform must have 3 values: [start, end, step]"
                raise ValueError(msg)
            
            cls_ = UniformIntegerParameterRange \
                   if isinstance(uniform[2], int) else \
                   UniformParameterRange
            hparam = cls_(name,
                          min_value=uniform[0], 
                          max_value=uniform[1], 
                          step_size=uniform[2])
        hparams.append(hparam)
    return hparams

    

def get_strategy(name:str):
    if name == "optuna":
        from clearml.automation.optuna import OptimizerOptuna
        # optimizer_kwargs
        # optuna_sampler (TPESampler by default)
        # optuna_pruner
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html#optuna.create_study
        return OptimizerOptuna
    elif name == "bohb":
        from clearml.automation.hpbandster import OptimizerBOHB
        return OptimizerBOHB
    elif name == "random":
        from clearml.automation import RandomSearch
        return RandomSearch
    elif name == "grid":
        from clearml.automation import GridSearch
        return GridSearch
    else:
        raise ValueError(f"Invalid strategy name '{name}'")

def get_sampler(sampler_cfg:dict):
    """
    See docs/hypertune.md sampler
    sampler_cfg (dict) : {
        "use": str
        "name": {**kwargs}
    }
    Returns:
        optuna.BaseSampler
        or
        None - default sampler
    """
    name = sampler_cfg["use"]
    if name == "default":
        return
    cfg = sampler_cfg[name]
    if name == "TPE":
        from optuna.samplers import TPESampler
        sampler = TPESampler(**cfg)
    elif name == "CMaEs":
        from optuna.samplers import CmaEsSampler
        sampler = CmaEsSampler(**cfg)
    elif name == "GP":
        from optuna.samplers import GPSampler
        sampler = GPSampler(**cfg)
    elif name == "NSGAII":
        from optuna.samplers import NSGAIISampler
        sampler = NSGAIISampler(**cfg)
    elif name == "NSGAIII":
        from optuna.samplers import NSGAIIISampler
        sampler = NSGAIIISampler(**cfg)
    elif name == "QMC":
        from optuna.samplers import QMCSampler
        sampler = QMCSampler(**cfg)
    elif name == "Grid":
        from optuna.samplers import GridSampler
        sampler = GridSampler(**cfg)
    elif name == "Random":
        from optuna.samplers import RandomSampler
        sampler = RandomSampler(**cfg)

    # other samplers
    else:
        raise ValueError(f"Invalid sampler '{name}'")
        
    return sampler

def get_pruner(pruner_cfg:dict):
    """
    See docs/hypertune.md pruner
    pruner_cfg (dict) : {
        "use": str
        "name": {**kwargs}
    }
    Returns:
        optuna.Basepruner
        or
        None - default pruner
    """
    name = pruner_cfg["use"]
    if name == "default":
        return
    cfg = pruner_cfg[name]
    if name == "Median":
        from optuna.pruners import MedianPruner
        pruner = MedianPruner(**cfg)
    elif name == "Percentile":
        from optuna.pruners import PercentilePruner
        pruner = PercentilePruner(**cfg)
    elif name == "SuccessiveHalving":
        from optuna.pruners import SuccessiveHalvingPruner
        pruner = SuccessiveHalvingPruner(**cfg)
    elif name == "Hyperband":
        from optuna.pruners import HyperbandPruner
        pruner = HyperbandPruner(**cfg)
    elif name == "Threshold":
        from optuna.pruners import ThresholdPruner
        pruner = ThresholdPruner(**cfg)
    elif name == "Wilcoxon":
        from optuna.pruners import WilcoxonPruner
        pruner = WilcoxonPruner(**cfg)
    elif name == "Nop":
        from optuna.pruners import NopPruner
        pruner = NopPruner(**cfg)

    # other pruner
    else:
        raise ValueError(f"Invalid pruner '{name}'")

    return pruner


def clearml_hypertune(
        base_task:str,
        hparams:Union[str, List[dict]]=HPARAMS_JSON,
        budget:int=20,
        config:Union[str, dict]="config.yaml",
        queue:str="default",
        comment:str=None,
        ):
    """
    Run hypertune with ClearML
    Args:
        base_task (str): this task will be cloned
        hparams (str, dict): List[dict] or JSON file
            [
                {
                    "name": "Args/batch_size",
                    "values": [5, 10, 12]
                },
                {
                    "name": "hparams/train/learning_rate",
                    "uniform": [0.01, 0.1, 0.03]
                },
                {
                    "name": "General/param",
                    "uniform": [0, 100, 20]
                }
            ]
        config (str, dict): /path/to/config.yaml
                            or hypertune params 
                            (see config.yaml:hypertune)
        queue (str): name of queue in ClearML
        comment (str): comment for hop run
    """
    # Load config
    if isinstance(config, str):
        # load config from yaml
        config_yaml = config
        config = utils.config_from_yaml(config_yaml)
        config = config["hypertune"]

    # Get experiment name of base task
    btask_id = base_task
    base_task = Task.get_task(btask_id)
    if not base_task:
        raise ValueError("Unknown task with given ID")
    project = base_task.get_project_name()
    # must be {ProjectName/experiment_name}
    try:
        experiment = project.split("/")[1]
    except IndexError:
        experiment = "experiment"

    # Get index of HOP
    exp_dir = osp.join(EXPERIMENTS_DIR, experiment)
    hop_name = utils.get_next_exprun(exp_dir, "hop-\d{3}")
    if comment:
        hop_name += '_' + comment
    hop_dir = osp.join(exp_dir, hop_name)

    # use given task as base if it has draft status
    # otherwise clone
    if base_task.get_status() != "created":  # draft
        # clone
        base_task = Task.clone(btask_id, name=hop_name + "-base")
    base_task.set_parameter("Args/task_name", hop_name)
    btask_id = base_task.id

    # HparamsOpt is also task
    Task.init(
        project_name='NoiseClassification/' + experiment,
        task_name=hop_name,
        task_type=Task.TaskTypes.optimizer,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
        auto_connect_streams=False,
        reuse_last_task_id=False
    )

    # apply config
    hparams = parse_clearml_hparams(hparams)
    titles, series, signs = [], [], []
    for obj, sign in zip(config["objective"], config["signs"]):
        title, ser = obj.split("/")
        titles.append(title)
        series.append(ser)
        signs.append(sign)

    if len(titles) > 1:
        raise NotImplementedError("More than 1 objective to optimize "\
                                  "is not implemented yet")
    else:
        titles, series, signs = titles[0], series[0], signs[0]

    sampler = get_sampler(config["sampler"])
    pruner = get_pruner(config["pruner"])
    optimizer_kwargs = dict()
    if sampler is not None:
        optimizer_kwargs["sampler"] = sampler
    if pruner is not None:
        optimizer_kwargs["pruner"] = pruner
    optimizer = HyperParameterOptimizer(
        base_task_id=btask_id,  
        hyper_parameters=hparams,  # what to optimize
        objective_metric_title=titles,
        objective_metric_series=series,
        objective_metric_sign=signs,  
        optimizer_class=get_strategy(config["alg"]),
        execution_queue=queue,  
        max_number_of_concurrent_tasks=config["concurency"],  
        total_max_jobs=budget,
        # for BOHB
        min_iteration_per_job=1,  
        max_iteration_per_job=150000,  
        **optimizer_kwargs
    )
    
    optimizer.set_report_period(config["report_period"])  # min

    optimizer.start(job_complete_callback=job_complete_callback)

    optimizer.wait()
    # now script is waiting

    top_exp = optimizer.get_top_experiments(top_k=5)
    for exp in top_exp:
        print(exp.name)

    # make sure background optimization stopped
    optimizer.stop()

    # save result to json
    top10 = optimizer.get_top_experiments_details(10)
    json_path = osp.join(hop_dir, "summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top10, f, ensure_ascii=False)


def optuna_hypertune(
        train_data:str,
        test_data:str,
        config:str='config.yaml',
        epochs:int=15,
        batch_size:int=500,
        cache_size:int=1000,
        gpu_id:int=0,
        experiment:str='experiment',
        log_step:int=1,
        comment:str=None,
        hparams:str=HPARAMS_JSON,
        budget:int=20,
):
    """
    train_data(str): path/to/train/data
    test_data(str): path/to/test/data
    config (str): /path/to/config.yaml
    experiment (str): experiment name
    cache_size (int): how much audio samples to store in RAM 
        for faster batch generation
    log_step (int): interval of loggoing step metrics
    comment (str): postfix for experiment run name
    """
    # load config
    config = utils.config_from_yaml(config)
    # load hypertune opt params
    hpo_params = config["hypertune"]
    target_metric = hpo_params["objective"]
    if len(target_metric) > 1:
        raise NotImplementedError("Multiobjective optimization "\
                                  "is not implemented yet")
    target_metric = target_metric[0]
    # "Metrics/train{test}_{name}" or "Loss/train{test}_{name}"

    # load hparams
    with open(hparams, 'rb') as f:
        hparams = json.load(f)  # list[dict]

    exp_dir = osp.join(EXPERIMENTS_DIR, experiment)
    hop_name = utils.get_next_exprun(exp_dir, "hop-\d{3}")
    if comment:
        hop_name += '_' + comment
    hop_dir = osp.join(exp_dir, hop_name)

    def objective(trial):
        # base setting
        trial_args = {
            "train_data": train_data,
            "test_data": test_data,
            "epochs": epochs,
            "batch_size": batch_size,
            "cache_size": cache_size,
        }
        trial_config = dict(config)  # copy
        
        for hparam in hparams:
            name = hparam["name"]
            if "values" in hparam:
                hp = trial.suggest_categorical(name, hparam["values"])
            else:
                start, end, step = hparam["uniform"]
                if isinstance(step, float):
                    hp = trial.suggest_float(name, start, end, step=step)
                else:
                    hp = trial.suggest_int(name, start, end, step=step)

            # override with hparams
            if name.startswith("Args/"):
                arg = name[5: ]
                assert arg in trial_args
                trial_args[arg] = hp
            elif name.startswith("hparams"):
                cfg = trial_config
                for key in name.split("/")[1 :-1]:
                    cfg = cfg[key]
                key = name.split("/")[-1]
                cfg[key] = hp

        metrics = train(
            **trial_args,
            config=trial_config,
            gpu_id=gpu_id,
            experiment=experiment,
            log_step=log_step,
            comment=comment,
            clearml=False,
            trial=trial,
            task_name=hop_name,
        )
        obj_value = utils.get_obj_value(metrics, target_metric)
        return obj_value

    # start hop
    sampler = get_sampler(hpo_params["sampler"])
    pruner = get_pruner(hpo_params["pruner"])
    direction = "maximize" if hpo_params["signs"][0].startswith("max") \
                else "minimize"
    optimizer_kwargs = {
        "direction": direction,

    }
    if sampler is not None:
        optimizer_kwargs["sampler"] = sampler
    if pruner is not None:
        optimizer_kwargs["pruner"] = pruner
    study = optuna.create_study(**optimizer_kwargs)
    study.optimize(objective, n_trials=budget)

    print("Best trial:")
    trial = study.best_trial

    print(f"  {target_metric}: ", trial.values[0])
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save summary
    summary = {
        "hyper_parameters": trial.params,
        "metrics": {target_metric: trial.values[0]}
    }
    json_path = osp.join(hop_dir, "summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False)


def hypertune(
        base_config:dict,
        budget:int=20
):
    def objective(trial):
        config = dict(base_config)  # copy
        config.update({
            "lr": trial.suggest_float("lr", 0.0001, 0.0001, step=0.0002),
            "n_layers": trial.suggest_int("n_layers", 3, 7, step=1),
            "opt": trial.suggest_categorical("opt", ["Adam", "SGD"])
        })
        # train multiple epochs and return best validation metric
        best_accuracy = train(config)
        return best_accuracy

    # Prunner & Sampler
    # see doc: https://github.com/Vladimetr/PyTorchMLProject/blob/noise-cls/docs/hypertune.md
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        direction="maximize",  # because accuracy
        pruner=pruner,
        sampler=sampler
    )
    study.optimize(objective, n_trials=budget)

    print("Best trial:")
    trial = study.best_trial

    print("  Values: ", trial.values[0])
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', '-hp', type=str, 
                    default=HPARAMS_JSON, 
                    help='Json file with hparams to tune')
    parser.add_argument('--budget', '-b', type=int, 
                        default=20, 
                        help='Number of experiments')
    parser.add_argument('--config', '-cfg', type=str, 
                        default='config.yaml', 
                        help='Config yaml with "hypertune" settings.')
    # train args
    parser.add_argument('--train-data', type=str, 
                        default='/app/data/esc50-5s-train.csv')
    parser.add_argument('--test-data', type=str, 
                        default='/app/data/esc50-5s-test.csv')
    parser.add_argument('--batch-size', '-bs', type=int, 
                        default=50)
    parser.add_argument('--gpu', type=int, dest="gpu_id", default=0,
                        help='which GPU to use')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--experiment', '-exp', type=str, 
                        default='experiment', 
                        help='Name of experiment')
    parser.add_argument('--cache-size', '-cs', type=int, 
                        default=1000,
                        help="how much audio samples to store in RAM"\
                             "for faster batch generation")
    parser.add_argument('--comment', '-m', type=str, default=None, 
                        help='Postfix for experiment run name')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    # Hypertune via ClearML
    parser.add_argument('--clearml', action='store_true', 
                    default=False, 
                    help='whether to use ClearML for experiment manager')
    parser.add_argument('--base-task', '-t', type=str, 
                        help='Base task ID in ClearML')
    parser.add_argument('--queue', '-q', type=str, 
                        default='default', 
                        help='Name of queue in ClearML')
    args = parser.parse_args()
    # Namespace to dict
    use_clearml = args.clearml

    if use_clearml:
        clearml_hypertune(
            base_task=args.base_task,
            budget=args.budget,
            hparams=args.hparams,
            config=args.config,
            queue=args.queue,
            comment=args.comment,
        )
    else:
        optuna_hypertune(
            train_data=args.train_data,
            test_data=args.test_data,
            config=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cache_size=args.cache_size,
            gpu_id=args.gpu_id,
            experiment=args.experiment,
            log_step=args.log_step,
            comment=args.comment,
            budget=args.budget,
        )
