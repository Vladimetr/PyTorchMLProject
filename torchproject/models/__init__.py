from typing import Union
from .base import BaseModel
from .tcn import TCNClassification


def init_model(model_cfg:Union[dict, None],
               preprocess_cfg:dict=None,
               training:bool=False,
               device:str="cpu",
                ) -> BaseModel:
    """
    model_cfg (dict): 
        {
            "{class_name}": kwargs (dict)
        }
    """
    model_cfg = dict(model_cfg)  # copy
    name = next(iter(model_cfg))
    params = model_cfg[name]
    weights = params.pop("weights")
    
    try:
        # define class
        model: BaseModel = globals()[name]
    except KeyError:
        raise ValueError(f"Invalid model name '{name}'")
    # try:
        # init
    model = model(preprocess_cfg=preprocess_cfg,
                  training=training,
                  device=device,
                  **params)
    # except TypeError:
    #     raise ValueError(f"Invalid model params {params}")

    if weights:
        model.load(weights)

    return model
