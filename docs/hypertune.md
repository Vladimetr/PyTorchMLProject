Hyper-Parameters Optimization (HOP)
=================
This is the process of searching best set of hyper-parameters (learning rate, batch size, dropout, data version, model version etc). Simplest ways are **Greedy search** of **Random search**. But more smart strategies exist, like TPE, BOHB.

### Implementation in ClearML
Now only one way is implemented - through [ClearML HPO](https://clear.ml/docs/latest/docs/fundamentals/hpo/). It has `optuna` and `hpbandster` libs in its core. 

1. Define base task. 
* Either completed one. In this case it will be cloned
* or created one (status draft). In this case `Args/task_name` will be overridden `hop-{X}`

Copy its `ID`

2. Define hypertune settings in `config.yaml:hypertune`. See `config.md` for more details

3. Define hyper-parameters to tune. Save in `hparams.json` </br>
Example
```
[
    {
        "name": "Args/batch_size",
        "values": [5, 10, 12]
    },
    {
        "name": "hparams/train/learning_rate",
        "uniform": [0.01, 0.1, 0.03]
    },
    {
        "name": "General/param",
        "uniform": [0, 100, 20]
    }
]
```
4. Run script (detached mode recommended for long process)
```bash
python3 -m noisecls.hypertune --base-task {ID} --hparams hparams.json --config config.yaml --queue default
```
- **--base-task/-t** - *ID of base task defined in (1)*
- **--hparams/-hp** - *hyper parameters to tune defined in JSON (3)*
- **--budget/-b** - *number of experiments*
- **--config/-cfg** - */path/to/config.yaml*
- **--queue/-q** - *queue name in ClearML*
- **--comment/-m** - *optional comment for hpo experiment*


#### Insufficient shared memory
After running experiment an error can occur: `DataLoader worker (pid(s) 1501) exited unexpectedly`. Set additional docker arg (extra arg) `--shm-size=64gb` to the base task.

### Sampler
Sampler is a strategy for choosing hparams set to run. Used when `alg = optuna`. [More details](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).
* [**TPE** (by default)](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler)
```yaml
sampler:
  use: TPE
  TPE:
    consider_prior: True
    prior_weight: 1.0
    # ...
```
or everything by default
```yaml
sampler:
  use: default
```

* [**CmaEs**](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler)
experimental
```yaml
sampler:
  use: CmaEs
  CmaEs:
    x0: null
    sigma0: null 
    # ...
```

* [**GP**](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html#optuna.samplers.GPSampler)
experimental
```yaml
sampler:
  use: GP
  GP:
    seed: null
    # ...
```

* [**NSGAII**](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html#optuna.samplers.NSGAIISampler)
```yaml
sampler:
  use: NSGAII
  NSGAII:
    population_size: 50
    # ...
```

* [**NSGAIII**](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html#optuna.samplers.NSGAIIISampler) 
experimental
```yaml
sampler:
  use: NSGAIII
  NSGAIII:
    population_size: 50
    # ...
```

* [**QMC**](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html#optuna.samplers.QMCSampler)
experimental
```yaml 
sampler:
  use: QMC
  QMC:
    qmc_type: "sobol"
    # ...
```


### pruner
pruner is an algorithm for early stopping experiments (trials). It happens based on target metric reports, i.e. every epoch values. Used when `alg = optuna`. [More details](https://optuna.readthedocs.io/en/stable/reference/pruners.html)

* [**Median** (by default)](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner)
```yaml
pruner:
  use: Median
  Median:
    n_startup_trials: 5
    n_warmup_steps: 0
    # ...
```
or everything by default
```yaml
pruner:
  use: default
```

* [**Percentile**](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html#optuna.pruners.PercentilePruner)
```yaml
pruner:
  use: Percentile
  Percentile:
    percentile : 15  # required [0..100]
    # ...
```

* [**SuccessiveHalving**](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner)
```yaml
pruner:
  use: SuccessiveHalving
  SuccessiveHalving:
    min_resource: "auto"
    reduction_factor: 4
    # ...
```

* [**Hyperband**](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner)
```yaml
pruner:
  use: Hyperband
  Hyperband:
    min_resource: 1
    max_resource: "auto"
    # ...
```

* [**Threshold**](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html#optuna.pruners.ThresholdPruner)
```yaml
pruner:
  use: Threshold
  Threshold:
    # either lower or upper must be defined
    lower: 0.21
    upper: null
    # ...
```

* [**Wilcoxon**](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html#optuna.pruners.WilcoxonPruner)
```yaml
pruner:
  use: Wilcoxon
  Wilcoxon:
    p_threshold: 0.1
    # ...
```

