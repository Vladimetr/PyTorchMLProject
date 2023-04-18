#!/bin/bash

docker run -it --rm \
    --name dev-torch-container \
    --network="host" \
    -v /mnt:/mnt \
    -w $PWD/../ \
    --user 1018:1018 \
    --gpus "device=8" \
    nlp_image:managers \
    /bin/bash