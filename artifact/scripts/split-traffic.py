import os
import sys

from functools import partial
from multiprocessing.pool import Pool
from subprocess import Popen, PIPE

NUM_PROCS = 4

def map_device_mac(macs_path):
    """
    Read the text file with device names and mac addresses (separated by commas)
    Create a map between mac address and device name, with mac addresses as key
    """
    mac_addrs = {}
    fh = open(macs_path, "r")
    for line in fh.readlines():
        device_name = "_".join(line.split(",")[0].strip().split(" "))
        mac_addrs[line.split(",")[1].strip()] = device_name.lower()
    
    return mac_addrs


def remove_empty(file_path):
    command = ["tshark", "-r", file_path]
    # Call Tshark on packets
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    # Get output. Give warning message if any
    out, err = process.communicate()
    if err:
        print("Error {} reading file: '{}'".format(err.decode('utf-8'), file_path))
    if len(out) == 0:
        os.remove(file_path)


def split_traffic(mac_addrs, output_dir, file):
    """
    Split the traffic in the pcap file using the mac addresses
    mapping.
    """
    if file.endswith(".pcap"):
        print("Reading file : {}".format(file))
        file_path = os.path.join(dir_path, file)
        for mac in mac_addrs:
            # Get the device information
            device = mac_addrs[mac]
            device_path = os.path.join(output_dir, device)
            os.makedirs(device_path, exist_ok=True)
            # Get the output file path
            output_file = os.path.join(device_path, device + "_" + "-".join(file.split("-")[1:]))
            if os.path.exists(output_file):
                continue
            # Filter to filter out the traffic for the given mac addr
            filter = "eth.src == {} or eth.dst == {}".format(mac, mac)
            # tshark command
            command = ["tshark", "-r", file_path,
                        "-Y", filter,
                        "-w", output_file]
            
            # Call Tshark on packets
            process = Popen(command, stdout=PIPE, stderr=PIPE)
            # Get output. Give warning message if any
            out, err = process.communicate()
            if err:
                print("Error reading file: '{}'".format(err.decode('utf-8')))
            
            remove_empty(output_file)


def main(dir_path, device_list_paths):
    """
    """

    output_dir = dir_path.rstrip("/") + "_per_device"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mac_addrs = map_device_mac(device_list_paths)
    print(mac_addrs)

    file_list = os.listdir(dir_path)
    file_list.sort()

    pool = Pool(processes=NUM_PROCS)
    func = partial(split_traffic, mac_addrs, output_dir)

    pool.map(func, file_list)
    # for file in file_list:
    #         split_traffic(mac_addrs, output_dir, file)    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR! Requires paths to the dataset directory and the device mac addresses as arguments")
        print("USAGE: python split-traffic.py <dataset-path> <device-mac-addrs-path>")
        exit(1)

    dir_path = sys.argv[1]
    macs_path = sys.argv[2]

    if not os.path.exists(dir_path):
        print("ERROR! Path to the dataset directory doesn't exist : {}".format(dir_path))
        exit(1)
    if not os.path.isdir(dir_path):
        print("ERROR! Path to the dataset is not a directory : {}".format(dir_path))
        exit(1)
    
    if not os.path.exists(macs_path):
        print("ERROR! Path to the text file with device mac addresses doesn't exist: {}".format(macs_path))
        exit(1)
    if not macs_path.endswith(".txt"):
        print("ERROR! Expected path to a text file with device mac addresses. Received: {}".format(macs_path.split(".")[-1]))
        exit(1)

    main(dir_path, macs_path)

