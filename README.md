## EXAMPLE USAGE ##
1. **Train** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --log-step 5 \
    --comment "my-train"
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

4. **Train experiment with manager** <br/>

4.1 **Train experiment with MLFlow** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --mlflow
```
>NOTE: this experiment must be created in MLFlow Server <br/>
>NOTE: add `-v $PWD/dev/mlflow/data:/mlflow/mlruns` to running container which is used for train.py and test.py

4.2. **Train experiment with ClearML** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --clearml
```
>NOTE: Credentials (access and secret keys) must be created in UI. And `/home/<user>/clearml.conf` must be defined:
```
api { 
    web_server: http://localhost:8080
    api_server: http://localhost:8008
    files_server: http://localhost:8081
    credentials {
        "access_key" = <ACCESS_KEY>
        "secret_key"  = <SECRET_KEY>
    }
}
```

4.3. **Train experiment with MLFlow and Tensorboard** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --mlflow \
    --tensorboard
```

4.4. **Train experiment with ClearML and Tensorboard** <br/>
```
python3 -m torchproject.train \
    --train-data data/train_manifest.csv \
    --test-data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --experiment 'my_experiment' \
    --clearml \
    --tensorboard
```


6. **Test** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --log-step 1 \
    --comment "my-test"
```

7. **Test with no saving** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --epochs 15 \
    --no-save
```

8. **Test with reference to train experiment run** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3
```
>NOTE: 'my_experiment/train/003*' must be created

9. **Test with reference to train experiment run and specific weights** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt
```

11. **Test with manager** <br/>

11.1. **Test with MLFlow** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt \
    --mlflow
```

11.2. **Test with ClearML** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt \
    --clearml
```
>NOTE: make things described in **4.2**

11.3. **Test with MLFlow and Tensorboard** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt \
    --mlflow \
    --tensorboard
```

11.4. **Test with ClearML and Tensorboard** <br/>
```
python3 -m torchproject.test \
    --data data/test_manifest.csv \
    --config config.yaml \
    --batch-size 50 \
    --experiment 'my_experiment' \
    --run 3 \
    --weights 5.pt \
    --clearml \
    --tensorboard
```

10. **Generate Tensorboard logs from metrics in csv** <br/>
```
python3 -m torchproject.utils -r 'my_experiment/test/001'
```
>NOTE: 'my_experiment/test/001*' must be created
