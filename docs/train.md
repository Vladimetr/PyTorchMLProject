TRAIN
===========
Each train run is new experiment that is saved in dir and manager. It's called `train/X_{comment}` or `train-X_{comment}` where **X** is incremental number of train run. </br>
All neccessary info is stored in dir and manager: 
- weights/;
- config.yaml;
- meta.yaml;
- train.csv;    *# step metrics on train data*
- test.csv;     *# step metrics on test data*
- summary.txt;  *# summary metrics*

See example in `dev/experiments/experiment/train/001/` </br>

> Files above allow to reproduce train experiment, e.g. re-run it on new data

Run
-----------
see `docker/train.sh`
```bash
docker run -d \
    ...
    $RUN_DOCIMAGE \
    python3 -m noisecls.train \
        --train-data /app/data/processed/esc50-5s-train.csv \
        --test-data /app/data/processed/esc50-5s-test.csv \
        -bs 5 \
        --epochs 8 \
        -exp my-exp \
        --comment "description" \
        --clearml
```

Parametres

<code> --config /path/to/config.yaml </code> </br>
*main config YAML for this project* </br>

<code> --train-data /path/to/train.csv </code> </br>
*path to train set CSV file (see example in data.md)*

<code> --test-data /path/to/test.csv </code> </br> 
*path to test set CSV file (see example in data.md)*

<code> --gpu 0 </code> </br>
*which gpu ID to use*

<code> --no-save </code> </br>
*without saving experiment neither manager nor dir*

<code> --experiment my_first_experiment </code> </br>
*name of experiment. This name is experiment dir and also in manager if it's used. For example (augmentation)*

<code> --clearml </code> </br>
*whether to log this experiment in ClearML*

<code> --comment my_first_train_run </code> </br>
*additional info (as postfix) for this experiment*

<code> --log-step 5 </code> </br>
*how often to log step metrics*

<code> --resume 32 </code> </br>
*train experiment run to resume training from last epoch (within given experiment)*

<code> --cache-size 1000 </code> </br>
*how much audio samples to store in RAM for faster batch generation*

How to
-------------

* #### Add metrics </br>
*Add metrics to `config:train:metrics` either from list in `noisecls/metrics.py` or losses*
```python
# metrics that can be computed on each step 
# they are based on confusion matrix
CM_METRICS = ["conf_matrix",
              "TP", "FN", "FP", "TN", 
              "acc", "recall", "precision" 
               ]
# metrics that are defined for particular class
CLASS_METRICS = ["TP", "FN", "FP", "TN",
                 "recall", "precision"
                 ]
# plots based on probs (not preds) and targets
# not able for step metrics
PLOTS = ["PR-curve", "ROC-curve"]
```

* #### Add optimizer </br>
*It's able to define another PyTorch optimizer.* 
```python
# Define optimizer
opt = train_params["opt"]
if opt == 'Adam':
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_params["learning_rate"], 
        weight_decay=train_params['weight_decay'])
else:
    raise Exception(f"No optimizer: '{opt}'")
```
*And set new name in `config:train:optimizer`*

* #### Add loss function </br>
*It's able to define another torch loss fucntion in `noisecls/metrics.py`.*</br>
*Note that name must ends with `*Loss`* </br>
*And set new name in `config:train:loss`*
```python
class Loss(metaclass=ABCMeta):
    """ Abstract Loss """
    @abstractmethod
    def __init__(self, device='cpu', *args, **kwargs):
        """
        Base parameters
        n_classes (int): number of classes
        device (str): 'cpu' or 'cuda'
        """
        pass

    @abstractmethod
    def __call__(self, pred:Tensor, targ:Tensor) -> dict:
        """
        B - batch size
        C - n classes
        Args:
            pred (B, C): predicted logits
            targ (B, C)): target one hot
        Returns:
            tuple:
              loss: object with method backward()
              dict: {'<NameLoss>': float}
        NOTE: loss object for backward is only one. 
        But dict can contain multiple key:values
        NOTE: <NameLoss> must end with '*Loss'
        """
        pass

class CrossEntropyLoss(Loss):
    def __init__(self, device='cpu', weights=None):
        if isinstance(weights, list):
            weights = Tensor(weights)
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.loss = self.loss.to(device)

    def __call__(self, pred: Tensor, targ: Tensor) -> tuple:
        loss = self.loss(pred, targ)
        ce_value = loss.item()  # float
        loss_values = {
            "CrossEntropyLoss": ce_value,
        }
        return loss, loss_values

def init_loss(loss_cfg:dict,
              device:str="cpu"
              ) -> Loss:
    """
    loss_cfg (dict): 
        {
            "{class_name}": kwargs (dict)
        }
    """
    loss_cfg = dict(loss_cfg)  # copy
    name = next(iter(loss_cfg))
    params = loss_cfg[name]
    try:
        # define class
        loss = globals()[name]
    except KeyError:
        raise ValueError(f"Invalid loss name '{name}'")
    try:
        # init
        loss = loss(**params, device=device)
    except TypeError:
        raise ValueError(f"Invalid loss params {params}")
    return loss
```

