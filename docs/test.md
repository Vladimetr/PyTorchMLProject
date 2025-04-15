TEST
===========
Each test run is new experiment too. It's also saved in dir and manager. </br>
* It's called `test/Y_{comment}` or `test-Y_{comment}` where **Y** is incremental number of test run. </br>
* If test refers to train run **X**, i.e. test run is based on train config, </br>
It's called `test/X-Y_{comment}` or `test-X-Y_{comment}` where **X** is number of train run. </br> 

All neccessary info is stored in dir and manager: 
- config.yaml;
- meta.yaml;
- test.csv;   *step metrics on test data*

See example in `dev/experiments/my_experiment/test/001-1/` </br>

> For example, it's allowed to test same model (from same train config) on different test data


Run
------------
see `docker/run.sh`
```bash
docker run -it --rm \
    ...
    $RUN_DOCIMAGE \
    python3 -m noisecls.eval \
        --data /app/data/processed/esc50-5s-test.csv \
        -bs 5 \
        -exp "my-exp" \
        --run-id 1 \
        --comment "some-test" \
        --clearml
```
Parameters

<code> --config /path/to/config.yaml </code> </br>
*main config YAML for this project* </br>

<code> --data /path/to/data.csv </code> </br>
*path to test set CSV file (see example in data.md)*

<code> --gpu 0 </code> </br>
*which gpu ID to use*

<code> --no-save </code> </br>
*without saving experiment neither manager nor dir*

<code> --experiment my_first_experiment </code> </br>
*name of experiment. This name is experiment dir and also in manager if it's used. For example (augmentation)*

<code>  --run-id 1 </code> </br>
*Optional reference to train run. It means that config will be loaded from this train experiment*

<code> --clearml </code> </br>
*whether to log this experiment in ClearML*

<code> --comment my_first_train_run </code> </br>
*additional info (as postfix) for this experiment*

<code> --log-step 5 </code> </br>
*how often to log step metrics*

<code> --cache-size 1000 </code> </br>
*how much audio samples to store in RAM for faster batch generation*


How to
-------------

* #### Add step metrics </br>
*Add metrics which will be calculated for each test batch to `config:test:step_metrics` either from list in `noisecls/metrics.py` or losses*
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

* #### Add summary metrics </br>
*Add metrics which will be calculated for whole test set to `config:test:sum_metrics` according to way above*
```python
METRICS = ["TP", "FN", "FP", "TN", 
           "acc", "recall", "precision", 
           "conf_matrix",
]
```

* #### Add plots </br>
*It's able to add plots like **ROC/Precision-Recall** curves, **confusion matrix** to manager. Set this to `config:test:plot*`*

