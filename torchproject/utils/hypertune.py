from typing import List, Union
import itertools


# NOTE: subparams are separated using '.'
tuned_params = {
    'vad.model.weights': [
        'weights1.pt',
        'weights2.pt',
        'weights3.pt',
    ]
    'vad.inference.pad': [1, 2],
    'vad.inference.threshold': [0.6, 0.7, 0.8],

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


# load base config
with open("config.yaml") as f:
    config = oyaml.load(f, Loader=oyaml.FullLoader)
base_config = OrderedDict(config)

combinations = get_combinations(tuned_params)
n_configs = len(combinations)
for i, combination in enumerate(combinations):
    print(f"{i+1}/{n_configs} test ...")

    config = config_update_params(base_config, combination)

    test(config)
