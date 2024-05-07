from .abstract import MLModel, IterativeModel, NonIterativeModel


def model_init(model_cfg:dict, train=False, device="cpu") -> MLModel:
    """
    Args:
        model_cfg (dict): required fields
            {'class': str, 'fdim': int, 'n_classes': int}
        train (bool): train mode or eval mode
    """
    model_cfg = dict(model_cfg)  # copy

    weights = model_cfg.pop("weights", None)
    model_class = model_cfg.pop("class")
    if model_class == 'random_forest':
        from .classic import RandomForest
        model = RandomForest(**model_cfg)
    elif model_class == 'dummy':
        from .abstract import DummyModel
        model = DummyModel(**model_cfg)
    
    # another model
    else:
        raise ValueError(f"Invalid model '{model_class}'")
    
    if not (isinstance(model, IterativeModel) or
            isinstance(model, NonIterativeModel)):
        raise ValueError("Model must belong either to Iterative "\
                         "or NonIterative class")
        # otherwise train process is not defined
    
    # Load weights
    if weights:
        model.load(weights)
    model.train_mode() if train else model.eval_mode()
    model.to_device(device)
    return model
