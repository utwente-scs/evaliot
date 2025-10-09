import os
import sys
import torch

from sklearn.model_selection import StratifiedKFold

from framework.model_training import ModelTraining


class YourSmartHomeTraining(ModelTraining):

    def __init__(self, config, train_data, cross_validation):
        """
        Sets the local variables from the config values

        config: config values in dict form
        """
        super().__init__(config["model-training"], train_data, cross_validation)
        self.config = config

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[YourSmartHome training] : Imported YourSmartHome code from the repo")
            
            from src.constants import WIN_SIZE
            from src.traffic_process import preprocess_traffic
            from src.util import load_device_file, encode_labels

            self.logger.debug("[YourSmartHome training] : Window size set to {}".format(str(WIN_SIZE)))
            
            # Load the file mapping mac addresses to devices
            self.paths["device-file"] = self.paths["device-file"].format(self.train_data["path"])
            self.paths["model-dir"] = self.paths["model-dir"].format(self.train_data["name"])
            self.device_mac_map = load_device_file(self.paths["device-file"])

            # Get the list of values from the map i.e the list of devices
            self.device_list = list(self.device_mac_map.values())
            self.logger.info("[YourSmartHome training] : Device-mac_address mapping file loaded in memory.")
            
            # Get the label encoder and label mapping by encoding the device list into integers
            self.labelencoder, self.label_mapping = encode_labels(self.device_list)
            
            # Get the list of pcaps from the dataset dir
            dataset_pcap_list = self.train_data["file-list"]

            self.dataset_base_path = os.path.join(self.paths["eval-dir"], train_data["name"])
            self.model_base_path = os.path.join(self.paths["model-dir"], train_data["name"])

            if not os.path.exists(self.dataset_base_path):
                os.makedirs(self.dataset_base_path)

            if not os.path.exists(self.paths["model-dir"]):
                os.makedirs(self.paths["model-dir"])

            self.dataset_base_path = os.path.join(self.dataset_base_path, train_data["name"])
                        
            # Preprocess the pcap files to get the features and the labels
            self.dataset_x, self.dataset_y = preprocess_traffic(
                                                                self.device_mac_map,
                                                                dataset_pcap_list,
                                                                self.dataset_base_path
                                                            )
            self.logger.info("[YourSmartHome training] : Finished loading and preprocessing the data.")

            # Select CUDA option if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Using the device: {}".format(self.device))

        except Exception as e:
            self.logger.exception("[YourSmartHome training] : {}".format(e))
            self.logger.error("[YourSmartHome training] : ERROR importing the YourSmartHome code from the repo")


    def run(self):
        """
        Run the code to train the model using Your smart home code.
        """
        try:
            from src.train_test_model import train_lstm_model

            if self.cross_validation:
                # If true, generate classifiers for every fold of the cross validation.
                # Store the classifiers in a the directory named after the fold index in the model directory
                self.x_train = []
                self.y_train = []
                self.x_test = []
                self.y_test = []
                
                self.model_list = []

                index = 0
                # Declare the stratified k fold object
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
                # Loop through the different folds
                for train_index, test_index in skf.split(self.dataset_x, self.dataset_y):
                    self.logger.info("[YourSmartHome training] : Starting Fold number: {}".format(str(index)))
                    # split the dataset into train and test dataset using the indices
                    self.x_train.append(self.dataset_x[train_index])
                    self.y_train.append(self.dataset_y[train_index])
                    self.x_test.append(self.dataset_x[test_index])
                    self.y_test.append(self.dataset_y[test_index])
                    
                    # Get the encoded labels for the training dataset
                    y_train = self.labelencoder.transform(self.y_train[index].ravel())
                    if self.config["model-training"]["bidirectional"]:
                        model_path = self.model_base_path + "-bidir-lstm-model-" + str(index) + ".sav"
                    else:
                        model_path = self.model_base_path + "-lstm-model-" + str(index) + ".sav"
                    # Train the LSTM model
                    self.model_list.append(
                                                train_lstm_model(
                                                    self.x_train[index],
                                                    y_train,
                                                    self.label_mapping,
                                                    model_path,
                                                    bidirectional=self.config["model-training"]["bidirectional"],
                                                    device=self.device
                                                )
                                            )
                                        
                    self.logger.info("[YourSmartHome training] : Finished training the model for the fold number: {}".format(str(index)))
                    self.logger.info("[YourSmartHome training] : Model is saved in the location: {}".format(model_path))

                    index += 1

            else:
                if self.config["model-training"]["bidirectional"]:
                    model_path = self.model_base_path + "-bidir-lstm-model.sav"
                else:
                    model_path = self.model_base_path + "-lstm-model.sav"
                
                dataset_y = self.labelencoder.transform(self.dataset_y.ravel())

                # Train the LSTM model
                self.model = train_lstm_model(
                                                self.dataset_x,
                                                dataset_y,
                                                self.label_mapping,
                                                model_path,
                                                bidirectional=self.config["model-training"]["bidirectional"],
                                                device=self.device
                                            )
                
                self.logger.info("[YourSmartHome training] : Finished training the model!")
                self.logger.info("[YourSmartHome training] : Model is saved in the location: {}".format(model_path))

            return False

        except Exception as e:
            self.logger.exception("[YourSmartHome training] : {}".format(e))
            return True


    def get_models(self):
        """
        Retreives the models trained by Your smart home code.
        """
        try:
            self.logger.debug("[YourSmartHome training] : Returning the trained models from the memory")
            # If the variable classifier_list hasn't been declared yet, it will throw an error
            if self.cross_validation:
                self.logger.debug("[YourSmartHome training] : The number of models in memory: {}".format(len(self.model_list)))
                return self.model_list
            else:
                return self.model

        except:
            # We will not use the models saved in files when cross validation is true.
            # As in that case the training and testing will happen in continuation 
            # and train and test indices will change every run.
            self.logger.debug("[YourSmartHome training] : Loading the trained models saved as file.")
            
            from src.object.lstm_model import Config, LstmModel

            config = Config()
            output_dim = len(self.device_list)

            model = None
            if self.config["model-training"]["bidirectional"]:
                model_path = self.model_base_path + "-bidir-lstm-model.sav"
            else:
                model_path = self.model_base_path + "-lstm-model.sav"

            if os.path.exists(model_path):
                model = LstmModel(config, output_dim, self.config["model-training"]["bidirectional"], self.device)
                model.load_state_dict(torch.load(model_path))
                return model
            else:
                self.logger.error("[YourSmartHome training] : Trained models not found in the expected location: {}"
                                    .format(model_path))
                exit(1)
