from typing import List, Union
import itertools
import argparse
from . import utils
from . import train


# NOTE: subparams are separated using '.'
tuned_params = {
    'model.cnn.n_layers': [2, 3],
    'loss.cross_entropy.pos_weight': [[0.8, 0.2], [0.6, 0.4]],

}


def get_combinations(params:dict) -> List[dict]:
    """
    in:
    {
        'a': [1, 2],
        'b': [3, 4],
    }
    out:
        [{'a': 1, 'b': 3},
         {'a': 1, 'b': 4},
         {'a': 2, 'b': 3},
         {'a': 2, 'b': 4}
        ]
    """
    # defined variable params
    keys, var_params = [], []
    for k, v in params.items():
        keys.append(k)
        var_params.append(v)

    combs_params = list(itertools.product(*var_params))
    # [(1, 3), (1, 4), (2, 3), (2, 4)]
    combinations = []
    for comb in combs_params:
        # (1, 3) -> {'a': 1, 'b': 3}
        combination = dict()
        for i, k in enumerate(keys):
            combination[k] = comb[i]
        combinations.append(combination)

    return combinations


def config_update_params(config:dict, params:dict) -> dict:
    """
    """
    config = dict(config)  # copy
    for param, value in params.items():
        subparams = param.split('.')
        subconfig = config
        try:
            for subparam in subparams[ :-1]:
                subconfig = subconfig[subparam]
            p = subparams[-1]
            subconfig[p]  # check if key exists
            subconfig[p] = value
        except KeyError:
            raise ValueError(f"Invalid param '{param}'")
    return config


def main(train_data:str,
         test_data:str,
         config:Union[str, dict]='config.yaml',
         epochs:int=15,
         batch_size:int=20,
         experiment:str='experiment',
         use_manager:bool=False,
         tensorboard:bool=False,
         log_step:int=1,
    ):
    """
    """
    # Load base config
    if isinstance(config, str):
        # load config from yaml
        config = utils.config_from_yaml(config)

    base_config = config
    combinations = get_combinations(tuned_params)
    n_configs = len(combinations)
    for i, combination in enumerate(combinations):
        print(f"{i+1}/{n_configs} Train...")

        config = config_update_params(base_config, combination)

        train.main(train_data=train_data,
             test_data=test_data,
             config=config,
             epochs=epochs,
             batch_size=batch_size,
             experiment=experiment,
             use_manager=use_manager,
             tensorboard=tensorboard,
             log_step=log_step,
             comment=f'hypertune-{i+1}'
        )
        utils.dict2yaml(config, f"config{i+1}.yaml")
        

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
    parser.add_argument('--experiment', '-exp', type=None, 
                        default='experiment', 
                        help='Name of existed MLFlow experiment')
    parser.add_argument('--manager', '-mng', action='store_true', 
                        dest='use_manager', default=False, 
                        help='whether to use ML experiment manager')
    parser.add_argument('--tensorboard', '-tb', action='store_true', 
                        default=False, 
                        help='whether to use Tensorboard')
    parser.add_argument('--log-step', '-ls', type=int, default=1, 
                        help='interval of log metrics')
    args = parser.parse_args()
    # Namespace to dict
    args = vars(args)

    main(**args)
