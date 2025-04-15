# env file by default
def_envfile="../.env"
envfile="${ENV_FILE:-$def_envfile}"

#!/usr/bin/env
. $envfile

docker run -it --rm \
    --shm-size=32gb \
    --name $TRAIN_NAME \
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
        python3 -m noisecls.train \
            --train-data /app/data/esc50-5s-train.csv \
            --test-data /app/data/esc50-5s-test.csv \
            -bs 5 \
            --epochs 8 \
            -exp my-exp \
            --comment "description" \
            --clearml
