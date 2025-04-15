ClearML manager
=======================

**ClearML** is tool for train/test experiment management as well as dataset organization

### Installation
1. Create new credentials in ClearML UI and paste generated config (tokens and IPs) to `clearml.conf`
2. Install clearml (if it's not installed) to **docker container** using pip and run `clearml-init`. Config must be located in `/home/user/clearml.conf`. Otherwise copy it from `dev/clearml.conf`
3. Update credentials in `config.yaml`

### Run experiments using agent
ClearML provides to run experiments from UI. These experiments will performed inside docker container. ClearML agent run containers according to given GPUs and queue.
1. Create new queue (or use default one) in ClearML UI
2. Stop previous worker (if needed) and create new one
```
clearml-agent list
```
```
clearml-agent daemon --stop {WORKER_ID}
```
```
clearml-agent daemon --queue {QUEUE_NAME} --detached --docker {image_name:tag} --gpus 0,1
```
It's able to create multiple workers on the same machine and GPU. But worker names will be conflicted by default. Create worker with custom name:
```
CLEARML_WORKER_ID="super:gpu4-1" clearml-agent daemon --queue {QUEUE_NAME} --detached --docker {image_name:tag} --gpus 4
```
>NOTE: These workers may listen to one queue and execute tasks in parallel if these tasks were added to this queue.

3. Make sure that your last commit is pushed to remote repo. Otherwise there will be an error. I didn't figure out how to overcome this issue.
4. Having successfull experiment, clone it. Cloned one has status **draft**
- 4.1. Make sure **container -> image** and **container -> arguments** are set
```
image: {IMAGE:TAG}
arguments: -v /mnt:/mnt -v /.../noise_classification/:/app/ -v /../:/data -w /app/ --user {UID}:{GID}
```
- 4.2 Update required params for new experiment. For example, cmd args, config, hyperparams

#### Permission denied
If you run experiment using clearml-agent (via worker&queue) you can face with problems with permissions to specific directories. In these cases check your `clearml.conf` on host. There must be paths like `/home/.clearml/` or `~/.clearml` `~/.cache`. Other directories like `/root/...` or `/home/{another_user}/` can cause problems to permissions. Make sure that `/home/.clearml/` is accessable to your user.
>NOTE: paths like `~/...` in `clearml.conf` refer to `/home/{container_user}/...` after running container.

#### Insufficient shared memory
After running experiment clearml-agent (via worker&queue) an error can occur: `DataLoader worker (pid(s) 1501) exited unexpectedly`. Set additional docker arg (extra arg) `--shm-size=64gb`.

#### Parameters setting
Note that conflicts may appear. For example, param `n_layers` in **config.yaml** may differ from one in **hyperparameters**. So that's why
* In tab Execution **SOURCE CODE -> SCRIPT PATH** will override those in **Configuration -> Args**
* Params in **hyperparameters** will override those in **сonfig.yaml**
* Changes **config.yaml** are made in **CONFIGURATION OBJECTS** (under **HYPERPARAMETERS**). In order to give another path to **config.yaml** whole experiment must be restarted.

- 3.3 After editing all required params, enqueue experiment to our queue

#### Sync with git remote
ClearML agent can run tasks in 2 scenarios:
1) one script can be executed without git repo
2) more than one script will only be executed when synchronised with a **git repo**. It means that current commit must be in git remote

#### Creating docker image for agent
When experiment is run using ClearML agent first time, a lot of libs must be installed. So it's recommended when all libs are installed and source code starts to implement, commit docker container to new image. And use this new image for furhter running ClearML experiments with agent. Set new image name and tag in experiment draft **container -> image**.

### ClearML server migration
If IP adress of ClearML server was changed, artifacts might not be reachable anymore. I didn't find other ways to solve this problem except to replace URL in source lib.
in `/home/user/.local/lib/python3.8/site-packages/clearml/datasets/dataset.py` change line
```
remote_url=task.artifacts[cls.__state_entry_name].url
```
to
```
remote_url=task.artifacts[cls.__state_entry_name].url.replace("{OLD_URL}", "{NEW_URL}"),
```
