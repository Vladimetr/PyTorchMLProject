from .base import Model


def model_init(name:str, params:dict, train=False, device="cpu") -> Model:
    weights = params.pop("weights")
    if name == 'cnn':
        from .cnn import CnnClassificator
        model = CnnClassificator(**params)
    
    # another model
    else:
        raise ValueError(f"Invalid model '{name}'")
    
    # Load weights
    if weights:
        model.load(weights)
    model.train() if train else model.eval()
    model = model.to(device)
    return model
