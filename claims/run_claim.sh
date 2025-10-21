#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Run the script with the Claim number. Example: ./run_claim.sh 1"
fi

# Get the path to the current directory
CURRENT_DIR="$(pwd -P)"

# Get the claim number from the argument.
CLAIM="$1"

CLAIM_DIR="${CURRENT_DIR}/claim${CLAIM}"
echo $CLAIM_DIR

# Get the list of files in the claims directory
mapfile -t FILE_LIST < <(find "$CLAIM_DIR" -type f -name "*.yml" -print | sort)

cd "../artifact"
pwd
# Iterate over the sorted file list
for FILE_PATH in "${FILE_LIST[@]}"; do
    echo "Running for Config File: $FILE_PATH"
    python main.py $FILE_PATH
done