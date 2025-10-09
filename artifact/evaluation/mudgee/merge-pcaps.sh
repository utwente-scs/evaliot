#!/bin/sh

if [ $# -lt 2 ]
  then
    echo "Need 2 arguments:"
    echo "path to the Result file"
    echo "Path to the directory containing pcap files"
    exit
fi

result=$1
directory=$2

mergecap $directory/*.pcap -w $result