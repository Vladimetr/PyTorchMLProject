Config
=========
`config.yaml` is main config file that defines whole project. There separate blocks for setting train, test, model configuration.

```yaml
classes:
  - mixer
  - game
  - user
```
*What classes are used for classificator. Labels in data must be from this list*

## model
```yaml
model:
  class: random_forest
  max_depth: 3
```
- **class** - *class (type) of model. According this class specific model is initialized in `blockchain_ml/models/__init__.py`*
- **kwargs** - *This parameters are used in `__init__` for exactly this model*

## preprocess
```yaml
preprocess: null
    # not used yet
```
*What algorithm is used to convert input data to tensor during **inference***

## train
```yaml
metrics:
  - acc
  - TP
  - FN
  - FP
pretrained: null
# only for iterative training
loss:
  class: cross_entropy
opt: Adam
learning_rate: 0.001
weight_decay: 0.005
grad_norm: 1.0
```
- **metrics** - *list of metrics to compute during train iterations. See available metrics in `blockchain_ml/metrics.py:METRICS`*
- **pretrained** - *path to model weights as pretrained*
- **loss** - *loss (with class and kwargs) for iterative training. See `blockchain_ml/metrics.py:init_loss`*
- **opt** - *Name of oprimizer for iterative training*
- **learning_rate**
- **weight_decay** - *Value for weight regularization*
- **grad_norm** - *Value for clipping gradients*

## test
```yaml
test:
  step_metrics:  # only scalars
    - acc
    - TP
    - FN
    - FP
  # for total test set
  sum_metrics:
    - acc
    - precision
    - recall
    - conf_matrix
  # via manager
  plot_conf_matrix: False
  plot_pr: False   # Precision-Recall curve
  plot_roc: False  # ROC curve
```
- **step_metrics** - *list of metrics to compute during test iterations. See available metrics in `blockchain_ml/metrics.py:METRICS`*
- **sum_metrics** - *list of metrics to compute for whole test set. See available metrics in `blockchain_ml/metrics.py:METRICS`*
- **plot_conf_matrix** - *whether to plot summary confusion in ClearML manager*
- **plot_pr** - *whether to plot summary Precision-Recall curve in ClearML manager*
- **plot_roc** - *whether to plot summary ROC curve in ClearML manager*

## ClearML manager
```yaml
manager:
  clearml:
    key_token: ***
    secret_token: ***
    subproject: True  # experiment format
  hparams:
    model:
      class: random_forest
      max_depth: 3
    train:
      opt: Adam
```
- **key_token** - *authorization token is defined in UI*
- **secret_token** - *authorization token is defined in UI*
- **subproject** - *how to store experiment format: in subdir or using _*
- **hparams** - *important parameters which can be changed in UI easily. They override config settings*
