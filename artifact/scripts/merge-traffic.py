#!/bin/python
import gc
import os
import sys
from datetime import datetime
from itertools import islice
from scapy.all import *

def batched(iterable, n):
    # from https://docs.python.org/3.12/library/itertools.html#itertools.batched
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def merge_files(device, list_files, output_dir):
    all_packets = []
    print("Merging {} PCAP files".format(str(len(list_files))))

    for pcap_file in list_files:
        packets = rdpcap(pcap_file)
        print("PCAP file read: {}".format(pcap_file))
        all_packets.extend(packets)

    avg_num_packets = int(len(all_packets)/len(list_files)) * 5
    
    sorted_packets = sorted(all_packets, key=lambda ts:ts.time)
    
    if avg_num_packets == 0:
        avg_num_packets = len(sorted_packets)

    if avg_num_packets > 0:    
        for batch in batched(sorted_packets, avg_num_packets):
            print(float(batch[-1].time))
            filename = device + "_" + datetime.fromtimestamp(float(batch[-1].time)).strftime("%m-%d-%Y_%H-%M-%S.%f") + ".pcap"
            filepath = os.path.join(output_dir, filename)
            print(filepath)
            if os.path.exists(filepath):
                continue
            wrpcap(filepath, batch)

    del(all_packets)
    del(sorted_packets)
    gc.collect()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error! Require 2 arguments: Input Directory and Output Directory!")
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print("Error! Given input directory is NOT a directory!")
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    list_dir = next(os.walk(input_dir))[1]
    list_dir = list(sorted(list_dir))

    print(list_dir)

    try:
        for device_dir in list_dir:

            input_device_path = os.path.join(input_dir, device_dir)
            output_device_path = os.path.join(output_dir, device_dir)
            
            os.makedirs(output_device_path, exist_ok=True)

            pcap_files = []
            for root, dirs, files in os.walk(input_device_path):
                if root is not input_device_path:
                    for file in files:
                        if file.endswith(".pcap") or file.endswith(".pcapng"):
                            pcap_files.append(os.path.join(root, file))
            num_files = len(pcap_files)
            if  num_files > 150:
                for index in range(0, num_files, 100):
                    max_index = index+100 if (index+100) < num_files else num_files
                    merge_files(device_dir, pcap_files[index:max_index], output_device_path)
            else:
                merge_files(device_dir, pcap_files, output_device_path)
    
    except Exception as e:
        print(e)


# ../datasets/MonIOTr/us/mix/roku-tv/roku-tv_04-27-2019_03-26-00.766472.pcap - error?