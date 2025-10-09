import numpy as np
import os
import pandas as pd
import pickle
import sys

from sklearn.model_selection import StratifiedKFold

from framework.model_training import ModelTraining

class DevicemeinTraining(ModelTraining):

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

            # Append the path to the Devicemein repository to the system paths
            # This allows for importing functions and classes from the repository like a library
            sys.path.append(self.paths["repo"])
            from extract_features import extract_flows, get_flow_windows
            from util import load_device_file

            self.logger.debug("[Devicemein training] : Imported Devicemein code from the repo")

            self.dataset_base_path = os.path.join(self.paths["eval-dir"], train_data["name"])
            self.device_mac_map = load_device_file(self.paths["device-file"].format(self.train_data["path"]))

            # Get the list of values from the map i.e the list of devices
            self.device_list = list(self.device_mac_map.values())

            if not os.path.exists(self.dataset_base_path):
                os.makedirs(self.dataset_base_path)

            if not os.path.exists(self.paths["model-dir"]):
                os.makedirs(self.paths["model-dir"])
            
            features_file = os.path.join(self.dataset_base_path, "features.pickle")
            labels_file = os.path.join(self.dataset_base_path, "labels.pickle")
            
            if os.path.isfile(features_file) and os.path.isfile(labels_file):
                self.dataset_x = pickle.load(open(features_file, 'rb')) 
                self.dataset_y = pickle.load(open(labels_file, 'rb')) 

            else:
                flows = extract_flows(self.train_data["file-list"])
                self.logger.debug("[Devicemein training] : Extracting features from the traffic flows")

                dataset = get_flow_windows(flows, self.device_mac_map)
                
                self.dataset_x, dataset_y = zip(*dataset)
                self.dataset_y = pd.DataFrame(dataset_y)
        
                self.logger.debug("[Devicemein training] :  Saving the extracted features into pickle files.")
                # Save the dataframes to pickle files    
                pickle.dump(self.dataset_x, open(features_file, "wb"))
                pickle.dump(self.dataset_y, open(labels_file, "wb"))
            
            # print(self.dataset_y.tolist())
            self.dataset_x = np.array(self.dataset_x, dtype=object)
            self.dataset_y = np.array(self.dataset_y, dtype=object).squeeze()

            self.logger.debug("[Devicemein training] : Loaded the training dataset")
        
        except Exception as e:
            self.logger.exception("[Devicemein training] : {}".format(e))
            self.logger.error("[Devicemein training] : ERROR importing the Devicemein code from the repo or loading the training dataset")
    

    def run(self):
        """
        Run the code to train the model using Devicemein code.
        """
        try:
            from objects.kmeans_tf import KMeansTF
            from train_test import train_lstm_ae

            if self.cross_validation:
                # If true, generate classifiers for every fold of the cross validation.
                # Store the classifiers in a the directory named after the fold index in the model directory
                self.x_train = []
                self.y_train = []
                self.x_test = []
                self.y_test = []
                
                self.trained_models = []

                index = 0
                # Declare the stratified k fold object
                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1111)
                # Loop through the different folds
                for train_index, test_index in skf.split(self.dataset_x, self.dataset_y):
                    self.logger.info("[Devicemein training] : Starting Fold number: {}".format(str(index)))
                    # split the dataset into train and test dataset using the indices
                    x_train = self.dataset_x[train_index]
                    x_test = self.dataset_x[test_index]
                    
                    # Train the LSTM AE model
                    encoder = train_lstm_ae(x_train)
                    x_train_enc = encoder.reconstruct(x_train)
                    x_test_enc = encoder.reconstruct(x_test)

                    # x_train_enc = x_train_enc.detach()
                    # x_test_enc = x_test_enc.detach()

                    self.x_train.append(x_train_enc)
                    self.y_train.append(self.dataset_y[train_index])
                    self.x_test.append(x_test_enc)
                    self.y_test.append(self.dataset_y[test_index])

                    num_classes=len(np.unique(self.dataset_y))

                    classifier = KMeansTF(n_clusters=num_classes, seed=1234)
                    best_k = classifier.fit(x_train_enc, self.y_train[index])

                    self.logger.info("[Devicemein Training] : Identified number of clusters: {}".format(str(best_k)))
                    
                    self.trained_models.append(classifier)

                    model_path = os.path.join(self.paths["model-dir"], self.train_data["name"] + "_" + str(index) + ".sav")
                    pickle.dump(classifier, open(model_path, 'wb'))
                            
                    self.logger.info("[Devicemein training] : Finished training the model for the fold number: {}".format(str(index)))
                    self.logger.info("[Devicemein training] : Model is saved in the location: {}".format(model_path))

                    index += 1

            else:
                # Train the LSTM AE model
                self.encoder = train_lstm_ae(self.dataset_x)
                model_path = os.path.join(self.paths["model-dir"], self.train_data["name"] + ".sav")
                
                if os.path.exists(model_path):
                    self.trained_models = pickle.load(open(model_path, 'rb'))
                    self.logger.info("[Devicemein training] : Model loaded from location: {}".format(model_path))
                else:
                    dataset_x_enc = self.encoder.reconstruct(self.dataset_x)
                    # dataset_x_enc = dataset_x_enc.detach()
                    # Train the LSTM model
                    num_classes=len(np.unique(self.dataset_y))

                    self.trained_models = KMeansTF(n_clusters=num_classes, seed=1234)
                    best_k = self.trained_models.fit(dataset_x_enc, self.dataset_y)
                    
                    self.logger.info("[Devicemein Training] : Identified number of clusters: {}".format(str(best_k)))

                    pickle.dump(self.trained_models, open(model_path, 'wb'))
                
                    self.logger.info("[Devicemein training] : Finished training the model!")
                    self.logger.info("[Devicemein training] : Model is saved in the location: {}".format(model_path))

            return False

        except Exception as e:
            self.logger.exception("[Devicemein training] : {}".format(e))
            return True

    def get_models(self):
        """
        """
        return self.trained_models