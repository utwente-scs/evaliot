#!/bin/sh

# Check if there was an argument in the command
if [ $# -lt 2 ]
  then
    echo "Need min 2 arguments: Path to the home directory of mudgee!"
    exit
else
  # Define the variable to store the directory containing all config files for MUDgee
  ROOT_DIR=$1
  DATASET=$2
  if [ $# -eq 2 ]
  then
    ITER=""
  else
    ITER=$3
  fi
fi

if [ -z "$ITER" ]
then
    CONFIG_DIR=$ROOT_DIR/"${DATASET}_configs"
else
    CONFIG_DIR=$ROOT_DIR/"${DATASET}_configs/${ITER}"
fi

EXEC_PATH="mudgee-1.0.0-SNAPSHOT.jar"

# Check if the given argument is a valid directory
if [ -d "$CONFIG_DIR" ]; then
    echo "Running MUDgee with the configs in directory: $CONFIG_DIR"

    # For every config in the directory, run the mudgee executable
    for FILE in $CONFIG_DIR/*
        do
            if [ -f "$FILE" ]; then
              java -jar "$EXEC_PATH" "$FILE"
            fi
        done
else
    echo "Path doesn't exist!"
fi

# Get the default result directory
RESULT_DIR=$ROOT_DIR/result
# If iteration exists, add it to the result directory to get the new result directory
if [ -z "$ITER" ]
then
    NEW_RESULT_DIR=$ROOT_DIR/"${DATASET}_result"
    mkdir -p $NEW_RESULT_DIR
    
    if [ -d "$NEW_RESULT_DIR/result" ]; then
        rm -rf "$NEW_RESULT_DIR/result"
    fi

else
    NEW_RESULT_DIR=$ROOT_DIR/"${DATASET}_result"
    mkdir -p $NEW_RESULT_DIR
    
    NEW_RESULT_DIR=$NEW_RESULT_DIR/"${ITER}"

    if [ -d "$NEW_RESULT_DIR" ]; then
        rm -rf "$NEW_RESULT_DIR"
    fi
fi


mv "$RESULT_DIR" "$NEW_RESULT_DIR"