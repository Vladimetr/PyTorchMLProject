# env file by default
def_envfile="../.env"
envfile="${ENV_FILE:-$def_envfile}"

#!/usr/bin/env
. $envfile

docker run -it --rm \
    --shm-size=64gb \
    --name $CONTAINER_NAME \
    --detach-keys="ctrl-t" \
    -v /mnt:/mnt \
    -v $PWD/../:/app/ \
    -v $DATA_DIR:/data \
    -w /app/ \
    -e CLEARML_KEY_TOKEN=$CLEARML_KEY_TOKEN \
    -e CLEARML_SEC_TOKEN=$CLEARML_SEC_TOKEN \
    -e AGENT_DOCIMAGE=$AGENT_DOCIMAGE \
    -e DOC_ARGS=$DOC_ARGS \
    --user $(id -u):$(id -g) \
    --gpus "device=$GPU_ID" \
    $RUN_DOCIMAGE \
        /bin/bash