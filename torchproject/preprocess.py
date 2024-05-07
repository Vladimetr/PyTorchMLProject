"""
Validation and preprocessing of raw audio
"""
from typing import Union
from collections import OrderedDict
import pandas as pd
import torch
from torch import Tensor
from . import utils


class BasePreprocess:
    """
    Extract vector for model from
    features dict
    """
    def __init__(self):
        pass

    def preprocess(self, features:OrderedDict) -> Tensor:
        """
        Vanila preprocess: extract values without preprocess
        Args:
            features (OrderedDict): Example {"num_addresses": 4789, ...}
        Returns:
            tensor: model input
        """
        values = list(features.values())
        x = torch.tensor(values)
        return x


# Define your own preprocess algorithm
# by inherit from BasePreprocess


def init_preprocessor(preprocess_params:Union[dict, None]
                            ) -> BasePreprocess:
    if preprocess_params is None:
        return BasePreprocess()
    # other preprocessors



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    preprocess_params:dict = config["preprocess"]

    data_path = 'data/processed/test.v1.csv'
    preprocessor = init_preprocessor(preprocess_params)

    data = pd.read_csv(data_path, index_col=False)
    for data_i in data.to_dict(orient='records'):
        model_inp = preprocessor.preprocess(data_i)
        print(model_inp)