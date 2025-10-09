#!/bin/bash

# Define the path for the reimplementations directory.
REIMPS_DIR="./reimplementations/"

declare -A list_reimps
list_reimps=( ['IOTSENTINEL']="https://github.com/ChakshuGupta/IoT-Sentinel-device-identification"
            ['MUDGEE']="https://github.com/ChakshuGupta/runtime-identification-mudgee"
            ['YOURSMARTHOME']="https://github.com/ChakshuGupta/your-smart-home-can-t-keep-a-secret"
            ['IOTDEVID']="https://github.com/ChakshuGupta/IoTDevIDv2"
            ['DEVICEMEIN']='https://github.com/ChakshuGupta/devicemien'
            ['GENIOTID']="https://github.com/ChakshuGupta/geniotid" )

# Create the directory and change the current directory
mkdir -p $REIMPS_DIR
cd $REIMPS_DIR


for item in "${!list_reimps[@]}"
do
    pwd
    echo "$item => ${list_reimps[$item]}"
    git clone ${list_reimps[$item]}
done