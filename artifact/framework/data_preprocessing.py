import logging
from scapy.all import *
from framework.packet import Packet

class DataPreprocessing:
    """
    
    """

    def __init__(self, config, traffic_type, input_datasets):
        """
        Sets local variables with the config values.
        
        [Args]
        config: config in dict format
        traffic_type: type of input traffic needed for testing the approach
        input_datasets: dict of all datasets with list of pcap files.
        """
        # Setup logger object
        self.logger = logging.getLogger(__name__)
        # Data format needed to test the model
        self.required_data_format = config["required-data-format"]
        # Use only known devices for testing (i.e. same devices in train and test dataset)
        self.use_known_devices = config["use-known-devices"]
        # Config of the dataset to be used for testing]
        self.test_dataset = config["test-dataset"]
        # Config of the dataset to be used for training
        self.train_dataset = config["train-dataset"]
        # Traffic type needed to train the model (per_device or mixed)
        self.traffic_type = traffic_type
        # List of config of the datasets
        self.input_datasets = input_datasets

    def preprocess_data(self, dataset):
        """
        Preprocess the input pcap files to change the input format to match
        the format required to test the approach.

        [Args]
        dataset: config of the dataset to process

        [Returns]
        processed_data: list or dict of data from the dataset
        """
        if "file-list" not in dataset:
            self.logger.error("[Data Preprocessing] : ERROR! Dataset {} doesn't have a file-list.".format(dataset["name"]))
            return
        # Process each .pcap file
        if dataset["type"] == "mixed":
            processed_data = []
            # For each pcap file, process it depending on the required data format
            # Creates a single list of all the packets
            for pcap_file in dataset["file-list"]:
                if self.required_data_format == "raw":
                    processed_data.extend(self.read_pcap(pcap_file))
                
                #######  Option to add other data formats here ########

                # elif self.required_data_format == "csv":
                #     processed_data.extend(self.convert_to_csv(pcap_file))
                # elif self.required_data_format == "json":
                #     processed_data.extend(self.convert_to_json(pcap_file))

        else:
            processed_data = {}
            # Read the list of files per device.
            for device_name in dataset["file-list"]:
                processed_data[device_name] = []
                # Creates a single list of all the packets for each device
                for pcap_file in dataset["file-list"][device_name]:
                    if self.required_data_format == "raw":
                        processed_data[device_name].extend(self.read_pcap(pcap_file))
                        
                    #######  Option to add other data formats here ########

                    # elif self.required_data_format == "csv":
                    #     processed_data[device_name].extend(self.convert_to_csv(pcap_file))
                    # elif self.required_data_format == "json":
                    #     processed_data[device_name].extend(self.convert_to_csv(pcap_file))
                    else:
                        continue
        
        return processed_data

    ###### Example Functions for handling other data formats ######
    
    # def convert_to_csv(self, pcap_file):
    #     """
    #     Convert pcaps to csv format
    #     TBD: implement
    #     """
    #     return []

    # def convert_to_json(self, pcap_file):
    #     """
    #     Convert pcaps to json format
    #     TBD: implement
    #     """
    #     return []

    def read_pcap(self, pcap_file):
        """
        Read pcap file and return the list of packets
        
        [Args]
        pcap_file: path to pcap file to read

        [Returns]
        packets: List of packets

        """

        self.logger.debug("[Data Preprocessing] : Reading PCAP file: ", pcap_file)
        # Read the raw packets using scapy
        raw_packets = rdpcap(pcap_file)

        packets = list()
        # Traverse through the list of raw packets and create packet objects
        for packet in raw_packets:
            packets.append(Packet(packet))
        
        return packets


    def get_dataset_info(self):
        """
        Retreive the dataset info corresponding to the train and test dataset
        in the input data

        [Returns]
        training_data: list of pcap files to be used for training
        testing_data: list of pcap files to be used for testing

        """
        train_dataset = None
        test_dataset = None

        for dataset in self.input_datasets:
            if dataset["name"] == self.test_dataset:
                test_dataset = dataset

            if dataset["name"] == self.train_dataset:
                train_dataset = dataset
        
        # Remove devices from test dataset if it doesn't exist in train dataset
        if self.use_known_devices and self.traffic_type == "per_device":
            self.logger.debug("[Data Preprocessing] : Using known devices is enabled!")
            for device in test_dataset["device-names"]:
                if device not in train_dataset["device-names"]:
                    test_dataset["device-names"].remove(device)
                    del test_dataset["file-list"][device]

            for device in train_dataset["device-names"]:
                if device not in test_dataset["device-names"]:
                    train_dataset["device-names"].remove(device)
                    del train_dataset["file-list"][device]
                    
            self.logger.debug("[Data Preprocessing] : Removed devices from test dataset that don't \
                              exist in train dataset and vice versa.")

        return train_dataset, test_dataset