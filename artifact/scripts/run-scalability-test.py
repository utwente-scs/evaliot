#!/bin/python

import numpy as np
import os
import shutil
import sys
import yaml

MIN_DEVICES = 7


def write_device_file(device_list, device_file):
    with open(device_file, "w") as f:
        f.writelines(device_list)


def load_device_file(device_file):
   """
   Load the mapping between devices and mac addresses
   """
   file_data = open(device_file, "r")
   device_mac_map = {}
   file_lines = file_data.readlines()
   for line in file_lines:
       if line.strip() == "":
           continue
       device = line.split(",")[0]
       mac = line.split(",")[1]
       device_mac_map[device.strip()] = mac.strip()

   return device_mac_map


def main(config_file):
    with open(config_file, 'r') as cfgfile:
        cfg = yaml.load(cfgfile, Loader=yaml.Loader)
    
    datasets = cfg["data-ingestion"]["list-datasets"]
    if len(datasets) > 1:
        print("ERROR! This test needs to run with only 1 dataset. More given.")
        exit(1)
    
    dataset_dir = datasets[0]["path"]
        
    if not os.path.exists(dataset_dir) and not os.path.isdir(dataset_dir):
        print("ERROR! Dataset path does not exist or it is not a directory!")
        exit(1)

    list_devices = os.listdir(dataset_dir)
    for device in list_devices:
        if not os.path.isdir(os.path.join(dataset_dir, device)):
            list_devices.remove(device)
    print("Dataset contains total {} number of devices.".format(str(len(list_devices))))

    # Create a temporary dir to store the device data
    root_dir = os.path.dirname(dataset_dir.rstrip("/"))
    tmp_dir = os.path.join(root_dir, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    list_devices.sort()
    # np.random.shuffle(list_devices)

    device_file = cfg["model-training"]["paths"]["device-file"].format(dataset_dir)
    device_map = load_device_file(device_file)
    shutil.copyfile(device_file, tmp_dir + "/devices.txt")

    iter = 0
    # Start the scalability test with MIN_DEVICES devices. Keep increasing the number of device with MIN_DEVICES
    # Until we have reached the max number of devices in the dataset directory
    while iter < len(list_devices):
        # This is to ensure the index never crosses the max length
        max_iter = min(iter + MIN_DEVICES, len(list_devices))
        # Get the list of devices to run the evaluation on
        subset_devices = list_devices[:max_iter]

        device_list = []
        for device in device_map:
            device_stripped = device.replace(" ", "_").lower()
            print(device, device_stripped)
            if device_stripped in subset_devices:
                device_list.append(device + " , " + device_map[device] + "\n")
        write_device_file(device_list, device_file)

        for device in list_devices:
            # The original path to the device data in the dataset dir
            org_path = os.path.join(dataset_dir, device)
            # The temporary path to the device data
            tmp_path = os.path.join(tmp_dir, device)
            # If the device in not in the subset, and it is still in the original location, move its data to the temp location
            if device not in subset_devices and os.path.exists(org_path) and os.path.isdir(org_path):
                shutil.move(org_path, tmp_dir)
            # If the device is in the subset and it is still in the temp location, move its data to the original dataset dir
            if device in subset_devices and os.path.exists(tmp_path):
                shutil.move(tmp_path, dataset_dir)
        
        # Delete any existing generated models
        model_dir = cfg["model-training"]["paths"]["model-dir"].format(cfg["data-preprocessing"]["train-dataset"])
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Check if the processed dataset exists and delete it
        processed_data_dir = os.path.join(cfg["model-training"]["paths"]["eval-dir"],
                                       cfg["data-preprocessing"]["train-dataset"])
        if os.path.exists(processed_data_dir):
            shutil.rmtree(processed_data_dir)
        
        # Run the evaluation with the selected subset of devices
        print("Running the evaluation with {} number of devices".format(len(subset_devices)))
        os.system("python3 main.py {}".format(config_file))

        iter = iter + MIN_DEVICES


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR! Require path to the config file as argument!")
        print("Usage: python3 run-scalability-test.py <path-to-config-file>")
        exit(1)
    
    config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print("ERROR! Incorrect path to the config file! File does't exist!")
        exit(1)
    if not config_file.endswith(".yml"):
        print("ERROR! Incorrect file type entered! Require a YAML file.")
        exit(1)

    main(config_file)