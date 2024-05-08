#!/bin/bash

docker run -d \
    --name train-antispoof \
    --network="host" \
    -v /mnt:/mnt \
    -v $PWD/../:/app/ \
    -w /app/ \
    --user $(id -u):$(id -g) \
    --gpus "device=8" \
    audio-image:latest \
    python3 -m torchproject.train \
        --epochs 15 \
        --clearml
        
