#!/usr/bin/env bash

set -e

if [[ -z "${INPUT_DATA}" ]]; then
    echo "INPUT_DATA environment variable is not defined"
    exit 1
fi

if [[ -z "${OUTPUT_DATA}" ]]; then
    echo "OUTPUT_DATA environment variable is not defined"
    exit 1
fi

python model_ecaas_agrifieldnet_silver/main_inferencing.py \
    --INPUT_DATA=${INPUT_DATA} \
    --OUTPUT_DATA=${OUTPUT_DATA} \
