Workers & Queues in ClearML
===========================
## Queues
Есть возможность добавлять эксперименты в очередь (queue) и ставить очередь на исполнение (worker-ом).
### 1. Создать свою очередь (у каждого пользователя будет своя очередь). В интерфейсе во вкладке `Workers&Queues -> Queues -> "+ NEW QUEUE"`
<br/>
<img src="img/new_queue.png"  width="90%" height="90%"> 

### 2. Имея завершенный без ошибок эксперимент склонировать его и задать имя.
**ВАЖНО:** данный коммит кода должен быть залит в удаленный репозиторий (must be *git pushed*)
<img src="img/clone.png"  width="90%" height="90%">
Склонированный эксперимент будет иметь статус **draft**. Эксперименты с таким статусом можно ставить в очередь на выполнение. 

### 3. В draft эксперименте поменяем нужные параметры (args, hparams, docker args и др.)
<img src="img/repo-info.png"  width="40%" height="40%">
<img src="img/change-params.png"  width="100%" height="100%">
Необходимо также заполнить поля конфигурации контейнера. Указать docker image и аргументы запуска, а именно `-v --user`. <br/>
**ВАЖНО:** GPU устанавливаются в другом месте <br/>
<img src="img/docker-args.png"  width="40%" height="40%">
<br/>
После чего ставим эксперимент в очередь на выполнение
<img src="img/enqueue.png"  width="90%" height="90%">
<br/>
эксперимент, находящийся в очереди на выполнение имеет статус **pending**

## Workers
Worker является исполнителем экспериментов в данной очереди. У worker-а в распоряжении есть список доступных GPU. 
<img src="img/agent.jpg"  width="80%" height="80%">
<br/>
Создать worker-а и указать для него очередь на выполнение можно двумя способами:
###  СПОСОБ 1
Создавать worker-а (исполнителя) каждый раз, передавая docker image и **номера GPU**
```
# delete previous worker
clearml-agent daemon --stop yakov_queue
# create new worker for queue "yakov_queue" with available GPU 0
clearml-agent daemon --queue yakov_queue --detached --docker yakov_image:latest --gpus 0
```
###  СПОСОБ 2
Создавать воркера (исполнителя) один раз, передав docker image и **все доступные GPU**
```
clearml-agent daemon --queue alex_queue --detached --docker alex_image:latest
``` 
А нужные GPU определять в скрипте train.py (или test.py)
```
python3 train.py --batch-size 100 ... --gpus 1,2
``` 
>ЗАМЕЧАНИЕ: таким способом можно изменять номера GPU в самом эксперименте (в графе args) и перезапускать обучение на других картах прям в интерфейсе - без консоли.
<img src="img/gpu-args.png"  width="100%" height="100%">

## Настройка параметров
ClearML позволяет редактировать в интерфейсе *аргументы командной строки, config.yaml, гиперпараметры*. Однако могут возникать конфликты. Например, параметр `n_layers` в **config.yaml** может иметь одно значение, а в **hyperparameters** - другое. Основываясь на работе в ClearML, а также на коде в данном репозитории существуют следующие правила:
* Во вкладе Execution в поле **SCRIPT PATH** аргументы запуска главнее, чем во вкладке Configuration в поле **Args**
* Параметры в **hyperparameters** главнее, чем в **сonfig.yaml**
* Изменение **config.yaml** происходит в поле **CONFIGURATION OBJECTS** (под **HYPERPARAMETERS**). Чтобы передать другой путь к **config.yaml** нужно перезапускать обучение через консоль.