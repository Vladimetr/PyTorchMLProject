from typing import Tuple
from abc import ABCMeta, abstractmethod
import pandas as pd
import torch
from torch import Tensor


class MLModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, fdim:int, n_classes:int,
                 training:bool=False):
        """
        Required params for all models
        fdim (int): number of features for classification 
                   (feature vector size)
        n_classes (int): number of classified classes
        training (bool): mode of model
        """
        pass
    
    def train_mode(self):
        pass

    def eval_mode(self):
        pass

    @abstractmethod
    def to_device(self, device:str):
        """ Batch will be on GPU """
        pass

    @abstractmethod
    def predict(self, x:Tensor):
        """
        B - batch size
        """
        pass

    @abstractmethod
    def save(self, weights_path:str):
        pass

    @abstractmethod
    def load(self, weights_path:str):
        pass


class IterativeModel(torch.nn.Module, MLModel,
                     metaclass=ABCMeta):
    """
    Back-prop based models
    using batch. 
    Model trains iteratively,
    each iteration produce metrics
    """
    @abstractmethod
    def __init__(self, fdim:int, n_classes:int,
                 training:bool=False, *args, **kwargs):
        super(IterativeModel, self).__init__(*args, **kwargs)
        self.train_mode() if training else self.eval_mode()

    def to_device(self, device:str):
        super().to(device)

    def train_mode(self):
        """ Torch train mode
        """
        super().train()

    def eval_mode(self):   
        """ Torch eval mode
        """
        super().eval()

    def init_weights(self, *args, **kwargs):
        """
        Set specific weights initialization
        """
        raise NotImplementedError()
    
    def predict(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        """
        B - batch size
        F - feature dim
        C - n classes
        Args:
            features (B, F): input features
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

    def load(self, weights) -> None:
        state_dict = torch.load(weights)
        self.load_state_dict(state_dict)

    def save(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)


class NonIterativeModel(MLModel, metaclass=ABCMeta):
    """
    Model is trained on whole dataset at once
    """
    @abstractmethod
    def __init__(self, fdim:int, n_classes:int,
                 training:bool=False, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, x:Tensor, y:Tensor):
        """ Fit whole dataset to train model
        M - datasize
        F - feature dim
        Args:
            x (M, F): inputs
            y (M, ): labels
        """
        pass



class DummyModel(IterativeModel):
    """ Just for test iterative training """
    def __init__(self, fdim:int, n_classes:int,
                 training:bool=False):
        super().__init__(fdim=fdim, n_classes=n_classes,
                         training=training)
        self.fc = torch.nn.Linear(fdim, n_classes)  
        self.act = torch.nn.Softmax(dim=1)
        self.train_mode() if training else self.eval_mode()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.float()
        logits = self.fc(x)
        probs = self.act(logits)       # (_, C)
        return logits, probs