# #!/bin/python
import json
import os
import sys

def update_config(config_path, pcap_path, new_path=None):
    """
    Update the mudgee config. Create a new file or update the existing one.
    """
    config_file = json.load(open(config_path))
    config_file["pcapLocation"] = pcap_path
    if new_path is not None:
        # Create a new config file with the file path
        json.dump(config_file, open(new_path, "w"), indent=4)
    else:
        # if new path is None, dump the json to the existing config file
        json.dump(config_file, open(config_path, "w"), indent=4)


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    if len(sys.argv) < 4:
        print("Need at leats 3 arguments: Root directory, name of the dataset and list of pcaps!")
        print("Usage(1): python update-config.py <root-dir> <dataset> <list-of-pcaps>")
        print("Usage(2): python update-config.py <root-dir> <dataset> <list-of-pcaps> <iter>")
        exit
    else:
        root_dir = sys.argv[1]
        dataset = sys.argv[2]
        list_files = json.loads(sys.argv[3])
        iter = None if len(sys.argv) == 4 else sys.argv[4].strip("-")
    
    # If iter is empty, replace with none
    iter = None if iter == "" else iter
    # Get the config directory
    config_dir = root_dir + "/" + dataset + "_configs"

    if iter is not None:
        # Create the iteration directory
        iter_dir = config_dir + "/" + iter
        # Create the config files for the iteration (only if the original config dir exists)
        if os.path.exists(config_dir):
            # Create the iteration directory
            if not os.path.exists(iter_dir):
                os.makedirs(iter_dir)
            # Create config files with the given pcap file locations
            for file in os.listdir(config_dir):
                if os.path.isdir(file):
                    continue
                if file.endswith(".json"):
                    device = file.split("_config")[0]
                    if device in list_files:
                        file_path = os.path.join(config_dir, file)
                        output_path = os.path.join(iter_dir, device + "_config.json")
                        update_config(file_path, list_files[device], output_path)
    
    else:
        # Check if the config fir exists
        if os.path.exists(config_dir):
            # Create config files with the given pcap file locations
            for file in os.listdir(config_dir):
                if file.endswith(".json"):
                    device = file.split("_config")[0]
                    if device in list_files:
                        file_path = os.path.join(config_dir, file)
                        # Update the pcap location in every config file
                        update_config(file_path, list_files[device])