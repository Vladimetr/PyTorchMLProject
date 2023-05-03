import pandas as pd
import torch
from typing import Tuple
from ..preprocess import BaseFeatures, init_features_extractor

class Model(torch.nn.Module):
    def __init__(self, features:dict=None):
        super().__init__()
        self.accumulated_grads = []
        if features:
            features_name = list(features.keys())[0]
            features_params = features[features_name]
            self.features = init_features_extractor(features_name,
                                                    features_params)
        else:
            self.features = None

    def init_params(self, *args, **kwargs):
        """
        Set specific weights initialization
        """
        raise NotImplementedError()
    
    def to(self, *args, **kwargs):
        if self.features:
            self.features = self.features.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        B - batch size
        T - time dim
        S - sample size
        F - feature dim
        C - n classes
        Args:
            features (B, F, T): input features
            or
            sample (B, 1, S): raw sample
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        if self.features:
            # (B, 1, S)
            x = self.features(x)
            # (B, F, T)
        return x

    def get_num_params(self):
        total_n = 0
        for name, p in self.named_parameters():
            # print(name, n)
            assert p.requires_grad
            n = p.numel()            
            total_n += n
        return total_n
    
    def validate_grads(self):
        """
        Check if all grads are not NaN. 
        Use it after backward()
        Raises:
            Exception: "Grad of '{LAYER}' is NaN
            Exception: "Grads not defined. Use backward() before"
        """
        for name, param in self.named_parameters():
            # print(name)
            try:
                grads = param.grad.data
            except AttributeError:
                msg = "Grads not defined. Use backward() before"
                raise Exception(msg)
            if torch.any(torch.isnan(grads)).item():
                raise Exception(f"Grad for param '{name}' is NaN")
            
    def get_grads(self) -> dict:
        """
        Get gradients summary - tuple(mean, std)
        per each layer
        Returns:
            dict: {'layer.name': ( mean(float), std(float) ) }
        """
        name_grads = dict()
        for name, param in self.named_parameters():
            assert param.requires_grad
            try:
                grads = param.grad.data
            except AttributeError:
                msg = "Grads not defined. Use backward() before"
                raise Exception(msg)
            mean = grads.mean().item()
            std = torch.std(grads).item()
            name_grads[name] = (mean, std)
            
        return name_grads 
    
    def accumulate_grads(self):
        grads = self.get_grads()
        self.accumulated_grads.append(grads)

    def grads2csv(self, csv_path:str):
        """
        Accumulated gradients to file.csv
        |--step--|--layer.name/mean--|--layer.name/std--|
        """
        data = []  # dicts
        for step_grads in self.accumulated_grads.items():
            new_step_grads = dict()
            for name, (mean, std) in step_grads.items():
                new_step_grads[name + '/mean'] = mean
                new_step_grads[name + '/std'] = std
            data.append(new_step_grads)
        
        df = pd.DataFrame.from_dict(data, orient='columns')
        df.to_csv(csv_path, sep=' ', index_label='step')

    def reset(self):
        self.accumulated_grads = []

    def load(self, weights) -> None:
        state_dict = torch.load(weights)
        self.load_state_dict(state_dict)

    def save(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)
