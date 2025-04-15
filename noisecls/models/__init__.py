from typing import Union
from .base import BaseModel


def init_model(n_classes:int,
               model_cfg:Union[dict, None],
               weights:str=None,
               training:bool=False,
               device:str="cpu",
                ) -> BaseModel:
    """
    model_cfg (dict): 
        {
            "preprocess": dict or None
            "use": "{class_name_i}",
            "{class_name_1}": 
                n_classes: int
                weights: str or None
                **kwargs
            "{class_name_2}": {...}
        }
    """
    preprocess_cfg = model_cfg.pop("preprocess", None)
    name = model_cfg["use"]
    params = model_cfg[name]

    print(f"Load model '{name}'...", flush=True)
    if name == "TCNClassification":
        from .tcn import TCNClassification
        model : BaseModel = TCNClassification
    elif name == "CLAPencoder":
        from .clap import CLAPencoder
        model : BaseModel = CLAPencoder
    elif name == "CLAPmultimodal":
        from .clap import CLAPmultimodal
        model : BaseModel = CLAPmultimodal
    elif name == "ResNext":
        from .resnext import ResNext
        model : BaseModel = ResNext
    elif name == "DynMobnet":
        from .mobnet import DynMobnet
        model : BaseModel = DynMobnet

    # define another model
    else:
        raise ValueError(f"Invalid model name '{name}'")
    
    # init
    try:
        model = model(n_classes=n_classes,
                      preprocess_cfg=preprocess_cfg,
                      training=training,
                      device=device,
                      **params)
    except TypeError:
        raise ValueError(f"Invalid model params {params}")

    if weights:
        model.load(weights)

    return model


if __name__ == '__main__':
    from ..utils import config_from_yaml
    import torch
    config = config_from_yaml('config.yaml')
    device = "cuda"

    model = init_model(n_classes=50, 
                              model_cfg=config["model"],
                              training=False,
                              device=device)

    # (BS, 1, S)
    samples = torch.rand((3, 1, 308700)).to(device=device)
    logits, probs = model(samples)
    print(logits.shape, probs.shape)
    # (BS, C)
