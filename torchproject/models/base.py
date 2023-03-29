import pandas as pd
import torch
from typing import Tuple

class Model(torch.nn.Module):
    def __init__(self):
        self.accumulated_grads = []
        super().__init__()

    def init_params(self, *args, **kwargs):
        """
        Set specific weights initialization
        """
        raise NotImplementedError()
    
    def forward(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        B - batch size
        T - time dim
        F - feature dim
        C - n classes
        Args:
            x (B, T, F): input
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        raise NotImplementedError("Abstract")

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