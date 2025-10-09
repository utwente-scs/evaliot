import numpy as np
import pandas as pd
import pickle
import os
import sys

from sklearn.utils import shuffle
from framework.model_testing import ModelTesting


class GenIoTIDTesting(ModelTesting):

    def __init__(self, config, trained_models, train_object, test_data):
        
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[GenIoTID testing] : Imported GenIoTID code from the repo")

            if not self.train_object.cross_validation:
                # This allows for importing functions and classes from the repository like a library
                from process_data import process_pcap, get_flows

                self.dataset_base_path = os.path.join(self.train_object.paths["eval-dir"], test_data["name"])

                if not os.path.exists(self.dataset_base_path):
                    os.makedirs(self.dataset_base_path)

                features_file = os.path.join(self.dataset_base_path, "features.pickle")
                labels_file = os.path.join(self.dataset_base_path, "labels.pickle")

                if os.path.isfile(features_file) and os.path.isfile(labels_file):
                    self.dataset_x = pickle.load(open(features_file, 'rb')) 
                    self.dataset_y = pickle.load(open(labels_file, 'rb')) 
                else:
                    dataset_all = pd.DataFrame()

                    for device, pcap_list in self.test_data["file-list"].items():
                        self.logger.debug("[GenIoTID testing] : Extracting packets from the files for device: {}".format(device))
                        # Send the pcap list to the process_pcap function and get packets
                        packets = process_pcap(pcap_list)
                        self.logger.debug("[GenIoTID testing] : Extracting features from the traffic flows")
                        # Extract device features from the packets and the device
                        device_features = get_flows(packets, device)
                        if device_features is None:
                            self.logger.warning("[GenIoTID testing] : Feature extraction failed for the device: {}".format(device))
                            continue
                        # If the dataset dataframe is empty, copy the features
                        if dataset_all.empty:
                            dataset_all = device_features
                        else:
                            # Else, append it to the exisiting dataset DataFrame
                            dataset_all = pd.concat([dataset_all, device_features], ignore_index=True)
                        
                    self.dataset_y = dataset_all["label"]
                    del dataset_all["label"]

                    tcp_flags = pd.DataFrame(dataset_all["tcp_flags"].to_list())
                    dns_queries = pd.DataFrame(dataset_all["dns_queries"].to_list())

                    del dataset_all["tcp_flags"]
                    del dataset_all["dns_queries"]

                    self.dataset_x = pd.concat([dataset_all, tcp_flags.add_prefix("tcp_flag_"), dns_queries.add_prefix("dns_query_")], axis=1)

                    self.logger.debug("[GenIoTID testing] : Saving the extracted features into pickle files.")
                    # Save the dataframes to pickle files    
                    pickle.dump(self.dataset_x, open(features_file, "wb"))
                    pickle.dump(self.dataset_y, open(labels_file, "wb"))

        except Exception as e:
            self.logger.exception("[GenIoTID testing] : {}".format(e))
            self.logger.error("[GenIoTID testing] : ERROR importing the GenIoTID code from the repo")

    
    def run(self):

        try:
            from train_test_model import test_model

            if self.train_object.cross_validation:
                for index in range(len(self.train_object.x_test)):
                    self.logger.debug("[GenIoTID testing] : Starting fold number {} of cross validation testing.".format(str(index)))

                    y_pred, y_prob = test_model(
                                                self.trained_models[index],
                                                self.train_object.x_test[index]
                                            )

                    self.true_values.append(self.train_object.y_test[index])
                    self.pred_values.append(y_pred)
                    self.prob_values.append(list(zip(self.train_object.y_test[index], y_pred, y_prob)))
                
                self.logger.debug("[GenIoTID testing] : Testing complete with cross validation")
            
            else:
                self.logger.debug("[GenIoTID testing] : Start testing with {} dataset".format(self.test_data["name"]))

                self.true_values = self.dataset_y
                train_col_names = self.train_object.dataset_x.columns
                test_col_names = self.dataset_x.columns.values

                # Verify the columns in the train and test dataset
                for col in test_col_names:
                    if col not in train_col_names:
                        del self.dataset_x[col]
                for col in train_col_names:
                    if col not in test_col_names:
                        self.dataset_x[col] = np.nan

                self.pred_values, y_prob = test_model(self.trained_models, self.dataset_x)

                if not self.config["data-preprocessing"]["use-known-devices"]:
                    for i in range(len(self.true_values)):
                        if y_prob[i] < self.config["data-preprocessing"]["threshold"]:
                            self.pred_values[i] = "Unknown"

                self.prob_values = list(zip(self.true_values, self.pred_values, y_prob))
                
                self.logger.debug("[GenIoTID testing] : Testing complete!")


        except Exception as e:
            self.logger.exception("[GenIoTID testing] : {}".format(e))
            return True
        