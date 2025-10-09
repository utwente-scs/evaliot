import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import sys

from framework.model_testing import ModelTesting


class DevicemeinTesting(ModelTesting):

    def __init__(self, config, trained_models, train_object, test_data):
        
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[Devicemein testing] : Imported Devicemein code from the repo")

            if not self.train_object.cross_validation:
                from extract_features import extract_flows, get_flow_windows
                from util import load_device_file

                self.dataset_base_path = os.path.join(self.train_object.paths["eval-dir"], test_data["name"])
                device_file = os.path.join(self.test_data["path"], "devices.txt")

                self.device_mac_map = load_device_file(device_file)
                print(self.device_mac_map)

                if not os.path.exists(self.dataset_base_path):
                    os.makedirs(self.dataset_base_path)

                features_file = os.path.join(self.dataset_base_path, "features.pickle")
                labels_file = os.path.join(self.dataset_base_path, "labels.pickle")

                if os.path.isfile(features_file) and os.path.isfile(labels_file):
                    self.dataset_x = pickle.load(open(features_file, 'rb')) 
                    self.dataset_y = pickle.load(open(labels_file, 'rb')) 
                else:
                    flows = extract_flows(self.test_data["file-list"])
                    self.logger.debug("[Devicemein testing] : Extracting features from the traffic flows")

                    dataset = get_flow_windows(flows, self.device_mac_map)
                    
                    self.dataset_x, dataset_y = zip(*dataset)
                    self.dataset_y = pd.DataFrame(dataset_y)
                    
                    self.logger.debug("[Devicemein testing] : Saving the extracted features into pickle files.")
                    # Save the dataframes to pickle files    
                    pickle.dump(self.dataset_x, open(features_file, "wb"))
                    pickle.dump(self.dataset_y, open(labels_file, "wb"))
                
                self.dataset_x = np.array(self.dataset_x, dtype=object)
                self.dataset_y = np.array(self.dataset_y, dtype=object).squeeze()

                self.logger.debug("[Devicemein testing] : Loaded the testing dataset")

        except Exception as e:
            self.logger.exception("[Devicemein testing] : {}".format(e))
            self.logger.error("[Devicemein testing] : ERROR importing the Devicemein code from the repo")


    def run(self):

        try:

            if self.train_object.cross_validation:
                for index in range(len(self.train_object.x_test)):
                    self.logger.debug("[Devicemein testing] : Starting fold number {} of cross validation testing.".format(str(index)))

                    y_pred, y_prob = self.trained_models[index].predict(
                                                                    self.train_object.x_test[index]
                                                                    )

                    self.true_values.append(self.train_object.y_test[index])
                    self.pred_values.append(y_pred)
                    self.prob_values.append(list(zip(self.train_object.y_test[index], y_pred, y_prob)))
                
                self.logger.debug("[Devicemein testing] : Testing complete with cross validation")
            
            else:
                self.logger.debug("[Devicemein testing] : Start testing with {} dataset".format(self.test_data["name"]))
                self.true_values = self.dataset_y.tolist()

                dataset_x_enc = self.train_object.encoder.reconstruct(self.dataset_x)
                # dataset_x_enc = dataset_x_enc.detach()
            
                if not self.config["data-preprocessing"]["use-known-devices"]:
                    self.pred_values, y_prob = self.trained_models.predict(dataset_x_enc)
                    for i in range(len(self.pred_values)):
                        if y_prob[i] < self.config["data-preprocessing"]["threshold"]:
                            self.pred_values[i] = "Unknown"

                else:
                    self.pred_values, y_prob = self.trained_models.predict(dataset_x_enc)

                self.prob_values = list(zip(self.true_values, self.pred_values, y_prob))
                
                self.logger.debug("[Devicemein testing] : Testing complete!")


        except Exception as e:
            self.logger.exception("[Devicemein testing] : {}".format(e))
            return True
        