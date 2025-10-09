import logging
import os
from scapy.all import *

class DataIngestion:
    
    def __init__(self, config, general_info):
        """
        Sets the local variables of the class using the input from the config file.

        [Args]
        config: config in dict format
        """
        self.logger = logging.getLogger(__name__)
        self.general_info = general_info
        self.list_datasets = config["list-datasets"]
        # Get the pcap files from the datasets listed in the config
        for val in range(len(self.list_datasets)):
            # if self.list_datasets[val]["load"] == False:
            self.get_device_names(val)
            self.get_pcaps_from_dir(val)
            self.logger.debug("[Data Ingestion] : Dataset {} loaded.".format(
                                                    self.list_datasets[val]["name"])
                                                )


    def get_device_names(self, index):
        """
        Get the list of devices in the dataset

        [Args]
        index: The index of the dataset, as specified in the list within the config file.
        """
        dataset = self.list_datasets[index]

        if not os.path.exists(dataset["path"]):
            self.logger.error("[Data Ingestion] : ERROR! Path to dataset {} doesn't exist."
                              .format(dataset["name"]))

        # Get the device names only if the dataset type is per_device
        # and the method's input type is also per_device
        if dataset["type"] == "per_device" and \
                        self.general_info["input-type"] == "per_device":
            dirs = next(os.walk(dataset["path"]))[1]
            dataset["device-names"] = dirs

        self.list_datasets[index] = dataset
        

    def get_pcaps_from_dir(self, index):
        """
        Search through the datasets directory and get the list of pcap files

        [Args]
        index: index number
        """
        dataset = self.list_datasets[index]

        if not os.path.exists(dataset["path"]):
            self.logger.error("[Data Ingestion] : ERROR! Path to dataset {} doesn't exist."
                              .format(dataset["name"]))

           
        # If the method's input type is "mixed", then input the dataset in this method
        # even if the dataset type is "per_device"
        if dataset["type"] == "mixed" or self.general_info["input-type"] == "mixed":

            for root, dirs, files in os.walk(dataset["path"]):
                #If the dataset name key does not exist in the dict, add it and create a 
                # new list.
                if "file-list" not in dataset: 
                    dataset["file-list"] = []
                # Add all pcap files to the "file-list" key in the dataset config
                for file in files:
                    if file.endswith(".pcap") or file.endswith(".pcapng"):
                        dataset["file-list"].append(os.path.join(root, file))
    
        else:
            if "file-list" not in dataset:
                dataset["file-list"] = {}
            
            for device in dataset["device-names"]:
                device_dir = os.path.join(dataset["path"], device)
                
                for root, dirs, files in os.walk(device_dir):
                    # If the device name key does not exist in the dict, create an empty list
                    if device not in dataset["file-list"]:
                        dataset["file-list"][device] = []
                    # Add all pcap files to the "file-list" key in the dataset config
                    for file in files:
                        if file.endswith(".pcap") or file.endswith(".pcapng"):
                            dataset["file-list"][device].append(os.path.join(root, file))
            
        self.list_datasets[index] = dataset