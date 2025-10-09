import os
import sys
import torch
import numpy as np

from framework.model_testing import ModelTesting


class YourSmartHomeTesting(ModelTesting):

    def __init__(self, config, trained_models, train_object, test_data):
        """
        Initialize a Your-smart-home testing class object with the following
        arguments:

        [Args]
        config: 
        model_dir:
        train_object:
        """
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[YourSmartHome testing] : Imported YourSmartHome code from the repo")
            
            from src.traffic_process import preprocess_traffic
            torch.multiprocessing.set_sharing_strategy('file_system')
            
            if not self.train_object.cross_validation:
                from src.util import load_device_file

                device_file = self.test_data["path"] + "/devices.txt"
                print(device_file)
                self.device_mac_map = load_device_file(device_file)
                print(self.device_mac_map)
                               
                if self.config["data-preprocessing"]["use-known-devices"]:
                    macs_to_remove = []
                    for mac, device in self.device_mac_map.items():
                        if device not in self.train_object.device_mac_map.values():
                            macs_to_remove.append(mac)
                    for mac in macs_to_remove:
                        del self.device_mac_map[mac]
                else:
                    # Get the list of values from the map i.e the list of devices
                    device_list = list(self.device_mac_map.values())
                    self.add_to_encoder(list(self.device_mac_map.values()))
                    print(device_list)
                    self.logger.info("[YourSmartHome testing] : Test data devices encoded. Number of devices: {}".format(len(device_list)))

                
                self.logger.info("[YourSmartHome testing] : Device-mac_address mapping file loaded in memory.")
                

                # Get the list of pcaps from the dataset dir
                dataset_pcap_list = self.test_data["file-list"]

                self.dataset_base_path = os.path.join(self.train_object.paths["eval-dir"], self.test_data["name"])
                
                if not os.path.exists(self.dataset_base_path):
                    os.makedirs(self.dataset_base_path)

                self.dataset_base_path = os.path.join(self.dataset_base_path, self.test_data["name"])


                self.logger.info("[YourSmartHome testing] : Base path to store the processed test data is: {}".format(self.dataset_base_path))
                            
                # Preprocess the pcap files to get the features and the labels
                self.dataset_x, self.dataset_y = preprocess_traffic(
                                                        self.device_mac_map,
                                                        dataset_pcap_list,
                                                        self.dataset_base_path
                                                    )
                self.logger.info("[YourSmartHome testing] : Finished loading and preprocessing the data.")


            # Select CUDA option if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Using the device: {}".format(self.device))

        except Exception as e:
            self.logger.exception("[YourSmartHome testing] : {}".format(e))
            self.logger.error("[YourSmartHome testing] : ERROR importing the YourSmartHome code from the repo")


    def add_to_encoder(self, device_list):
        """
        Check if there are devices not in the encoder.
        If not, add them to the encoder.

        """
        for device in device_list:
            if device not in self.train_object.labelencoder.classes_:
                new_label = len(self.train_object.labelencoder.classes_)
                self.train_object.labelencoder.classes_ = np.append(
                                                        self.train_object.labelencoder.classes_,
                                                        device
                                                        )
                self.train_object.label_mapping[device] = new_label


    def run(self):
        """
        Run the Your-smart-home Tests
        """
        try:
            from src.train_test_model import test_lstm_model

            try:
                # Get the train and test data here using in index
                if self.train_object.cross_validation:

                    for index in range(len(self.train_object.x_test)):
                        self.logger.debug("[YourSmartHome testing] : Starting fold number {} of cross validation testing.".format(str(index)))

                        y_test = self.train_object.labelencoder.transform(
                                                    self.train_object.y_test[index].ravel()
                                                )
                        self.logger.debug("[YourSmartHome testing] : Label values encode for fold number {}".format(str(index)))
                        # Test the Generated model
                        y_true, y_pred, y_prob = test_lstm_model(
                                                    self.trained_models[index],
                                                    self.train_object.x_test[index],
                                                    y_test,
                                                    self.train_object.labelencoder,
                                                    device=self.device
                                                )

                        self.true_values.append(y_true)
                        self.pred_values.append(y_pred)
                        self.prob_values.append(y_prob)
                    
                    self.logger.debug("[YourSmartHome testing] : Testing complete with cross validation")
                
                else:
                    self.logger.debug("[YourSmartHome testing] : Start testing with {} dataset".format(self.test_data["name"]))
                    print("Type of the Y dataset: " ,set(self.dataset_y))
                    # Encode the test labels using the labelencoder
                    dataset_y = self.train_object.labelencoder.transform(self.dataset_y.ravel())
                    self.logger.debug("[YourSmartHome testing] : Label values encoded.")
                    # Test the Generated model
                    self.true_values, self.pred_values, y_prob = test_lstm_model(
                                                                    self.trained_models,
                                                                    self.dataset_x,
                                                                    dataset_y,
                                                                    self.train_object.labelencoder,
                                                                    device=self.device
                                                                )
                    if not self.config["data-preprocessing"]["use-known-devices"]:
                        for i in range(len(self.true_values)):
                            if y_prob[i] < self.config["data-preprocessing"]["threshold"]:
                                self.pred_values[i] = "Unknown"
                                print(i, self.pred_values[i])
                    
                    self.prob_values = list(zip(self.true_values, self.pred_values, y_prob))
                    self.logger.info("[YourSmartHome testing] : Completed testing the model and the report is now saved in the json file.")


            except Exception as e:
                self.logger.exception("[YourSmartHome testing] : ".format(e))
                self.logger.error("[YourSmartHome testing] : ERROR! Unable to get the train and test data from the train object")

        except Exception as e:
            self.logger.exception("[YourSmartHome testing] : {}".format(e))
            return True
