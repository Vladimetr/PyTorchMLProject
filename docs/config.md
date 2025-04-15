Config
=========
`config.yaml` is main config file that defines whole project. There are separate blocks for setting train, test, model configuration.

```yaml
classes:
  - real
  - fake
```
*What classes are used for classificator. Labels in data must be from this list*

## model
```yaml
model:
  preprocess: dict or null
  <ClassModelName>:
    n_classes: int
    kwarg1: value
    kwarg2: value
```
- **preprocess** - dict (or null) with preprocess params. If defined preprocess will be inside nn.Module (on GPU)
- **class** - *class (type) of model. According this class specific model is initialized in `noisecls/models/__init__.py`*
- **kwargs** - *This parameters are used in `__init__` for exactly this model*

## preprocess
```yaml
preprocess: dict or null
    # not used yet
```
*What algorithm is used to convert input data to tensor as model input. If not defined, preprocess before model is not used*

>NOTE: use either preprocess inside model (may be on GPU) or before model (only on CPU)

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
  <ClassLossName>: dict
opt: Adam
learning_rate: 0.001
weight_decay: 0.005
grad_norm: 1.0
```
- **metrics** - *list of metrics to compute during train iterations. See available metrics in `noisecls/metrics.py:METRICS`*
- **pretrained** - *path to model weights as pretrained*
- **loss** - *loss (with class and kwargs). See `noisecls/metrics.py:init_loss`*
- **opt** - *Name of oprimizer for iterative training*
- **learning_rate**
- **weight_decay** - *Value for weight regularization*
- **grad_norm** - *Value for clipping gradients*

## evaluaion
```yaml
eval:
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
- **step_metrics** - *list of metrics to compute during test iterations. See available metrics in `noisecls/metrics.py:METRICS`*
- **sum_metrics** - *list of metrics to compute for whole test set. See available metrics in `noisecls/metrics.py:METRICS`*
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

## Hyper-parameters tuning (HPO)
```yaml
hypertune:
  alg: optuna
  # Loss/name, Metrics/name
  objective:
    - Loss/CrossEntropy
  # min/max for every metric above
  signs:
    - min_global
  concurency: 1
  time_limit: 1000  # min
  report_period: 3  # min
  sampler:
    use: default
  pruner:
    use: default

```
- **alg** - *algorithm (or strategy) to use: `optuna` | `bohb` | `random` | `grid`*
- **objective** - *loss or another metric to optimize. Title/name*
- **signs** - *`min_global` for Loss and Miss or `max_global` for Accuracy and Recall*
- **concurency** - *number of concurent tasks to execute*
- **time_limit** - *whole optimization process time limit (in min)*
- **report_period** - *how often data will be updated in main ClearML task*
- **pruner** - see full description in `docs/hypertune.md pruner`
- **sampler** - see full description in `docs/hypertune.md Sampler`

