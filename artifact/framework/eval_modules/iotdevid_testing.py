import numpy as np
import pandas as pd
import os
import sys

from sklearn.utils import shuffle
from framework.model_testing import ModelTesting


class IoTDevIDTesting(ModelTesting):

    def __init__(self, config, trained_models, train_object, test_data):
        
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[IoTDevID testing] : Imported IoTDevID code from the repo")

            if not self.train_object.cross_validation:
                from src.constants import FEATURE_DICT
                from src.feature_extraction import extract_features
                from src.util import load_device_file, list_files

                device_file = self.test_data["path"] + "/devices.txt"
                self.device_mac_map = load_device_file(device_file)

                if self.config["data-preprocessing"]["use-known-devices"]:
                    macs_to_remove = []
                    for mac, device in self.device_mac_map.items():
                        if device not in self.train_object.device_mac_map.values():
                            macs_to_remove.append(mac)
                    for mac in macs_to_remove:
                        del self.device_mac_map[mac]

                self.dataset_base_path = os.path.join(self.train_object.paths["eval-dir"], test_data["name"])

                # Get the list of pcaps from the dataset dir
                dataset_pcap_list = self.test_data["file-list"]

                if not os.path.exists(self.dataset_base_path):
                    os.makedirs(self.dataset_base_path)

                self.dataset_features_file = os.path.join(self.dataset_base_path, test_data["name"] + "_features.csv")

                if os.path.exists(self.dataset_features_file):
                    self.dataset = pd.read_csv(self.dataset_features_file, low_memory=False, dtype=FEATURE_DICT)
                else:
                    # Extract the features
                    for device in dataset_pcap_list:
                        self.logger.info("[IoTDevID testing] : Extracting features for the device : {}".format(device))
                        extract_features(dataset_pcap_list[device], self.device_mac_map)

                    self.dataset_csv_list = list_files(self.test_data["path"], ".csv")
                
                    self.dataset = pd.DataFrame()

                    cols = list(FEATURE_DICT.keys())
                    for file_name in self.dataset_csv_list:
                        df = pd.read_csv(file_name, low_memory=False, dtype=FEATURE_DICT)
                        df = df[cols]
                    
                        if self.dataset.empty:
                            self.dataset = df
                        else:
                            self.dataset = pd.concat([self.dataset, df])

                    self.dataset.to_csv(self.dataset_features_file, index=False)

        except Exception as e:
            self.logger.exception("[IoTDevID testing] : {}".format(e))
            self.logger.error("[IoTDevID testing] : ERROR importing the IoTDevID code from the repo")

    
    def run(self):
        try:
            from src.evaluation import test_model, merged

            step = 13
            mixed = True

            if self.train_object.cross_validation:
                for index in range(len(self.train_object.x_test)):
                    self.logger.debug("[IoTDevID testing] : Starting fold number {} of cross validation testing.".format(str(index)))
                    
                    m_test = self.train_object.x_test[index]["MAC"]
                    del self.train_object.x_test[index]["MAC"]
                    
                    # Test the Generated model
                    y_true = []
                    y_pred = []
                    y_prob = []

                    y_true_per_rep, y_pred_per_rep, y_prob_per_rep, _ = test_model(
                                                                        self.train_object.x_test[index],
                                                                        self.train_object.y_test[index],
                                                                        self.trained_models[index]
                                                                    )
                    if step != 1:
                        for i in range(len(y_pred_per_rep)):
                            y_pred_per_rep[i], _ = merged(m_test, y_pred_per_rep[i], step, mixed)
                            zero_indices = np.where(y_true_per_rep[i]=="0")[0]
                            y_true.extend(np.delete(y_true_per_rep[i], zero_indices))
                            y_pred.extend(np.delete(y_pred_per_rep[i], zero_indices))
                            y_prob.extend(np.delete(y_prob_per_rep[i], zero_indices))
                    else:
                        for i in range(len(y_pred_per_rep)):
                            zero_indices = np.where(y_true_per_rep[i]=="0")[0]
                            y_true.extend(np.delete(y_true_per_rep[i], zero_indices))
                            y_pred.extend(np.delete(y_pred_per_rep[i], zero_indices))
                            y_prob.extend(np.delete(y_prob_per_rep[i], zero_indices))

                    self.true_values.append(y_true)
                    self.pred_values.append(y_pred)
                    self.prob_values.append(list(zip(y_true, y_pred, y_prob)))
                
                self.logger.debug("[IoTDevID testing] : Testing complete with cross validation")

            else:
                self.logger.debug("[IoTDevID testing] : Start testing with {} dataset".format(self.test_data["name"]))

                self.dataset = shuffle(self.dataset, random_state=42)

                m_test = self.dataset["MAC"]
                del self.dataset["MAC"]

                dataset_x = self.dataset[self.dataset.columns[0:-1]]
                dataset_x = dataset_x.to_numpy()

                self.dataset[self.dataset.columns[-1]] = self.dataset[self.dataset.columns[-1]].astype('category')
                dataset_y = self.dataset[self.dataset.columns[-1]]

                y_prob = []

                # Predict the values
                y_true_per_rep, y_pred_per_rep, y_prob_per_rep, _ = test_model(dataset_x, dataset_y, self.train_object.trained_models)
                if step != 1:
                    for i in range(len(y_pred_per_rep)):
                        y_pred_per_rep[i], _ = merged(m_test, y_pred_per_rep[i], step, mixed)
                        zero_indices = np.where(y_true_per_rep[i]=="0")[0]
                        self.true_values.extend(np.delete(y_true_per_rep[i], zero_indices))
                        self.pred_values.extend(np.delete(y_pred_per_rep[i], zero_indices))
                        y_prob.extend(np.delete(y_prob_per_rep[i], zero_indices))
                else:
                    for i in range(len(y_pred_per_rep)):
                        zero_indices = np.where(y_true_per_rep[i]=="0")[0]
                        self.true_values.extend(np.delete(y_true_per_rep[i], zero_indices))
                        self.pred_values.extend(np.delete(y_pred_per_rep[i], zero_indices))
                        y_prob.extend(np.delete(y_prob_per_rep[i], zero_indices))
                
                if not self.config["data-preprocessing"]["use-known-devices"]:
                        for i in range(len(self.true_values)):
                            if y_prob[i] < self.config["data-preprocessing"]["threshold"]:
                                self.pred_values[i] = "Unknown"
                
                self.prob_values = list(zip(self.true_values, self.pred_values, y_prob))

                self.logger.debug("[IoTDevID testing] : Testing complete!")
        
        except Exception as e:
            self.logger.exception("[IoTDevID testing] : {}".format(e))
            return True