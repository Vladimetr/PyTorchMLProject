## EXAMPLE USAGE ##
1. **Train** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --log-step 5
```

2. **Train with no saving** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --no-save
```

3. **Train specific experiment with comment** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --comment 'example-run'
```

4. **Train experiment with MLFlow** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --manager
```
>NOTE: this experiment must be created in MLFlow Server
>NOTE: add `-v $PWD/dev/mlflow/data:/mlflow/mlruns` to run container

5. **Train experiment with MLFlow and Tensorboard** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --manager \
    --tensorboard
```

6. **Test** <br/>
```
python3 -m torchproject.train \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
```

7. **Test with no saving** <br/>
```
python3 -m torchproject.train \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --no-save
```

8. **Test with reference to train experiment run** <br/>
```
python3 -m torchproject.train \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --run 3
```
>NOTE: 'my_experiment/train/003*' must be created

9. **Test with reference to train experiment run and specific weights** <br/>
```
python3 -m torchproject.train \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt
```

10. **Generate Tensorboard logs from metrics in csv** <br/>
```
python3 -m torchproject.utils -r 'my_experiment/test/001'
```
>NOTE: 'my_experiment/test/001*' must be created
