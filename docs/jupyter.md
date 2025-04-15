Jupyter notebook allows you to run data processing and train scripts using interactive tools like buttons

## Run jupyter server
```bash
cd docker
sh jupyter_server.sh
```
Additionally you can change port and token in shell script

## Use jupyter kernel in VS code
1. Open `.ipynb` in VS code
2. Press `Select kernel` -> `Existed jupyter server`
3. Paste link of juputer server
4. Choose Python3

## Data Processing
`notebooks/data_process.ipynb` </br>
Functions in this notebook perform tasks from `noisecls/data` </br>

![structure image](/docs/img/jup_data.png)

### Additional arguments - kwargs
Some tasks like `filter-classes` require additional kwargs. In these cases you have to write them in line `kwargs` in JSON format/ It will be converted to dict. </br>
>Example: `{"drop_classes": ["dog", "cat"]}`

## Train
`notebooks/train.ipynb` </br>
This notebook can generate shell script `docker/train_{X}.sh` for running training with given params </br>

![structure image](/docs/img/jup_train.png)
