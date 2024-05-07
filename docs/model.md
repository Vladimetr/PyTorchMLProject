MODEL
===========

Each model can be represented as following `blockchain_ml/models/abstract.py`:
```python
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
```

There were 2 types of models under this project: 

- #### IterativeModel
*This PyTorch model is trained iteratively using batches, loss function, gradients calculation and backprop with multiple epochs.*

```python
class IterativeModel(MLModel):
    def __init__(self):
        pass

    def train_step()
        pass
```

- #### NonIterativeModel
*This model is trained on whole dataset at once. Like random forests or linear regression. It's like fit/predict approach.*

```python
class NonIterativeModel(MLModel):
    def __init__(self):
        pass
```

How to define new model
-----------
1. In order to add new model create new class by inheriting from either `IterativeModel` or `NonIterativeModel`. Dont't forget to define required methods according to described API. 
2. Add initialization to `blockchain_ml/models/__init__.py`
```python
if model_class == 'random_forest':
    from .classic import RandomForest
    model = RandomForest(**model_cfg)
elif model_class == 'dummy':
    from .abstract import DummyModel
    model = DummyModel(**model_cfg)

# your new model
else:
    raise ValueError(f"Invalid model '{model_class}'")
```
3. Set new model in `config.yaml` with **class** and **parametres**