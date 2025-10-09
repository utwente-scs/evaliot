#!/bin/sh

DATASET_DIR=$1
CONFIG_DIR=$2

if [ $# -eq 1 ]
then
    echo "Need min 2 arguments: Path to the dataset directory and path to the config directory!"
    exit
fi

if [ -d "$DATASET_DIR" ]; then
    if [ -d "$CONFIG_DIR" ]; then
        # For every config in the directory, run the mudgee executable
        for DIR in $DATASET_DIR/*
            do
                DIR_NAME=$(basename $DIR)
                touch $CONFIG_DIR/$DIR_NAME"_config.json"
            done
    else
        echo "Config Path doesn't exist!"
    fi
else
    echo "Dataset Path doesn't exist!"
fi