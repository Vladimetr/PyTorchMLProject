from .base import Model


def model_init(name:str, params:dict, train=False, device="cpu") -> Model:
    params = dict(params)  # copy
    weights = params.pop("weights")
    if name == 'tcn':
        from .tcn import TCNClassification
        model = TCNClassification(**params)
    
    # another model
    else:
        raise ValueError(f"Invalid model '{name}'")
    
    # Load weights
    if weights:
        model.load(weights)
    model.train() if train else model.eval()
    model = model.to(device)
    return model
