from typing import Tuple
from abc import ABCMeta, abstractmethod
# import pandas as pd
import torch
from torch import Tensor
from ..preprocess import init_preprocessor


class BaseModel(torch.nn.Module,
                metaclass=ABCMeta):
    def __init__(self, 
                 n_classes:int=2,
                 preprocess_cfg:dict=None,
                 device:str="cpu",
                 ):
        super(BaseModel, self).__init__()
        self.n_classes = n_classes
        if preprocess_cfg:
            preprocess_cfg = dict(preprocess_cfg)  # copy
            preprocess_cfg["device"] = device
            self.preprocessor = init_preprocessor(preprocess_cfg)
        else:
            self.preprocessor = None

    def to(self, device:str) -> None:
        if self.preprocessor is None:
            return
        self.preprocessor.to(device)

    def init_weights(self, *args, **kwargs):
        """
        Set specific weights initialization
        """
        raise NotImplementedError()
    
    def preprocess(self, x:Tensor) -> Tensor:
        """
        Returns:
            (B, F, T)
        """
        if self.preprocessor is None:
            return x
        return self.preprocessor(x)

    @abstractmethod
    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        """
        B - batch size
        F - feature dim
        T - time dim
        C - n classes
        Args:
            features (B, F, T): input features (already preprocessed)
            or
            samples (B, 1, S): raw samples (preprocess required)
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        pass

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

    def load(self, weights_path:str) -> None:
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)

    def save(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)
