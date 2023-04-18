#!/bin/bash

# sh create_clearml_worker --gpus 0
# sh create_clearml_worker --gpus 0,1

clearml-agent daemon --stop
clearml-agent daemon --queue default --docker nlp_image:managers --detached $1 $2