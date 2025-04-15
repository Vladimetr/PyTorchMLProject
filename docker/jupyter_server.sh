# env file by default
def_envfile="../.env"
envfile="${ENV_FILE:-$def_envfile}"

#!/usr/bin/env
. $envfile

docker run -d \
    -p $JUP_PORT:8888 \
    --name $JUP_NAME \
    -v /mnt:/mnt \
    -v $PWD/../:/app \
    -v $DATA_DIR:/data \
    --user $(id -u):$(id -g) \
    --gpus "device=$GPU_ID" \
    $JUP_IMAGE \
        jupyter notebook --NotebookApp.token="$JUP_TOKEN"

echo Go to http://localhost:$JUP_PORT/tree?token=$JUP_TOKEN