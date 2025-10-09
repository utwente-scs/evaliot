import os
import pandas as pd
import pickle
import sys

from sklearn.model_selection import StratifiedKFold

from framework.model_training import ModelTraining

class GenIoTIDTraining(ModelTraining):

    def __init__(self, config, train_data, cross_validation):
        """
        Sets the local variables from the config values

        config: config values in dict form
        """
        # Call the init function of the base model training class and set the 
        # values of the global variables.
        super().__init__(config["model-training"], train_data, cross_validation)
        
        try:
            # Format the model directory to get the dataset specific dir path to save the models
            self.paths["model-dir"] = self.paths["model-dir"].format(self.train_data["name"])

            # Append the path to the GenIoTID repository to the system paths
            # This allows for importing functions and classes from the repository like a library
            sys.path.append(self.paths["repo"])
            from process_data import process_pcap, get_flows

            self.logger.debug("[GenIoTID training] : Imported GenIoTID code from the repo")

            self.dataset_base_path = os.path.join(self.paths["eval-dir"], train_data["name"])
            self.model_base_path = os.path.join(self.paths["model-dir"], train_data["name"])
            if not os.path.exists(self.dataset_base_path):
                os.makedirs(self.dataset_base_path)

            if not os.path.exists(self.model_base_path):
                os.makedirs(self.model_base_path)
            
            features_file = os.path.join(self.dataset_base_path, "features.pickle")
            labels_file = os.path.join(self.dataset_base_path, "labels.pickle")
            
            if os.path.isfile(features_file) and os.path.isfile(labels_file):
                self.dataset_x = pickle.load(open(features_file, 'rb')) 
                self.dataset_y = pickle.load(open(labels_file, 'rb')) 

            else:
                dataset_all = pd.DataFrame()

                for device, pcap_list in self.train_data["file-list"].items():
                    self.logger.debug("[GenIoTID training] : Extracting packets from the files for device: {}".format(device))
                    # Send the pcap list to the process_pcap function and get packets
                    packets = process_pcap(pcap_list)
                    self.logger.debug("[GenIoTID training] : Extracting features from the traffic flows")
                    # Extract device features from the packets and the device
                    device_features = get_flows(packets, device)
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

                self.logger.debug("[GenIoTID training] : Saving the extracted features into pickle files.")
                # Save the dataframes to pickle files    
                pickle.dump(self.dataset_x, open(features_file, "wb"))
                pickle.dump(self.dataset_y, open(labels_file, "wb"))

            self.logger.debug("[GenIoTID training] : Loaded the training dataset")
        
        except Exception as e:
            self.logger.exception("[GenIoTID training] : {}".format(e))
            self.logger.error("[GenIoTID training] : ERROR importing the GenIoTID code from the repo or loading the training dataset")

    
    def run(self):
        
        try:
            from train_test_model import train_model

            if self.cross_validation:
                # If true, generate classifiers for every fold of the cross validation.
                # Store the classifiers in a the directory named after the fold index in the model directory
                self.x_train = []
                self.y_train = []
                self.x_test = []
                self.y_test = []
                
                self.trained_models = []

                # Declare the stratified k fold object
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
                index = 0
                # Loop through the different folds
                for train_index, test_index in skf.split(self.dataset_x, self.dataset_y):
                    self.logger.info("[GenIoTID training] : Starting Fold number: {}".format(str(index)))
                    # split the dataset into train and test dataset using the indices
                    self.x_train.append(self.dataset_x.iloc[train_index])
                    self.y_train.append(self.dataset_y.iloc[train_index])
                    self.x_test.append(self.dataset_x.iloc[test_index])
                    self.y_test.append(self.dataset_y.iloc[test_index])
                    
                    # Train the model using the data
                    model = train_model(self.x_train[index], self.y_train[index])

                    model_path = os.path.join(self.model_base_path, self.train_data["name"] + "_" + str(index) + ".sav")
                    pickle.dump(model, open(model_path, 'wb'))
                   
                    self.trained_models.append(model)

                    index += 1
                    self.logger.info("[GenIoTID training] : Finished training the model for the fold number: {}".format(str(index)))
                    self.logger.info("[GenIoTID training] : Model is saved in the location: {}".format(model_path))


            else:
                self.logger.debug("[GenIoTID training] : Start training with {} dataset".format(self.train_data["name"]))

                model_path = os.path.join(self.model_base_path, self.train_data["name"] + ".sav")
                
                if os.path.exists(model_path):
                    self.trained_models = pickle.load(open(model_path, 'rb'))
                else:
                    self.trained_models = train_model(self.dataset_x, self.dataset_y)
                    pickle.dump(self.trained_models, open(model_path, 'wb'))


        except Exception as e:
            self.logger.exception("[GenIoTID training] : {}".format(e))
            return True
    

    def get_models(self):
        """
        """
        return self.trained_models