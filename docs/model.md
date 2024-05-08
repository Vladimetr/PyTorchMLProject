MODEL
===========

Each model can be represented as following `torchproject/models/base.py`:
```python
class BaseModel(torch.nn.Module,
                metaclass=ABCMeta):
    def __init__(self, 
                 n_classes:int=2,
                 preprocess_cfg:dict=None,
                 device:str="cpu",
                 ):
        super(BaseModel, self).__init__()
        self.n_classes = n_classes
        # ...

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

    def load(self, weights_path:str) -> None:
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)

    def save(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)
```

### How to define new model
-----------
1. In order to add new model create new class by inheriting from `BaseModel`. Dont't forget to define required methods according to described API. 
2. Add initialization to `torchproject/models/__init__.py`
```python
preprocess_cfg = model_cfg.pop("preprocess", None)
name = next(iter(model_cfg))
params = model_cfg[name]

try:
    # define class
    model: BaseModel = globals()[name]
except KeyError:
    raise ValueError(f"Invalid model name '{name}'")
try:
    # init
    model = model(preprocess_cfg=preprocess_cfg,
                    training=training,
                    device=device,
                    **params)
except TypeError:
    raise ValueError(f"Invalid model params {params}")

if weights:
    model.load(weights)
```
3. Set new model in `config.yaml` with **class** and **parametres**