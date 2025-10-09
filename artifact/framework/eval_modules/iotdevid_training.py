import pandas as pd
import pickle
import os
import sys

from sklearn.model_selection import StratifiedKFold

from framework.model_training import ModelTraining


class IoTDevIDTraining(ModelTraining):

    def __init__(self, config, train_data, cross_validation):
        
        super().__init__(config["model-training"], train_data, cross_validation)
        self.config = config

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[IoTDevID training] : Imported IoTDevID code from the repo")

            from src.constants import FEATURE_DICT
            from src.feature_extraction import extract_features
            from src.util import load_device_file, list_files

            # Load the file mapping mac addresses to devices
            self.paths["device-file"] = self.paths["device-file"].format(self.train_data["path"])
            self.paths["model-dir"] = self.paths["model-dir"].format(self.train_data["name"])
            self.device_mac_map = load_device_file(self.paths["device-file"])

            # Get the list of pcaps from the dataset dir
            dataset_pcap_list = self.train_data["file-list"]
            
            self.dataset_base_path = os.path.join(self.paths["eval-dir"], train_data["name"])
            self.model_base_path = os.path.join(self.paths["model-dir"], train_data["name"])

            if not os.path.exists(self.dataset_base_path):
                os.makedirs(self.dataset_base_path)

            if not os.path.exists(self.model_base_path):
                os.makedirs(self.model_base_path)
            
            self.dataset_features_file = os.path.join(self.dataset_base_path, train_data["name"] + "_features.csv")

            if os.path.exists(self.dataset_features_file):
                self.dataset = pd.read_csv(self.dataset_features_file, low_memory=False, dtype=FEATURE_DICT)
            
            else:
                # Extract the features
                for device in dataset_pcap_list:
                    self.logger.info("[IoTDevID training] : Extracting features for the device : {}".format(device))
                    extract_features(dataset_pcap_list[device], self.device_mac_map)

                self.dataset_csv_list = list_files(train_data["path"], ".csv")
                
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
            self.logger.exception("[IoTDevID training] : {}".format(e))
            self.logger.error("[IoTDevID training] : ERROR importing the IoTDevID code from the repo")

    
    def run(self):
        
        try:
            from src.evaluation import train_model, ml_list         

            ml_algo = list(ml_list.keys())[0]

            if self.cross_validation:
                self.dataset_x = self.dataset[self.dataset.columns[0:-1]]
                self.dataset[self.dataset.columns[-1]] = self.dataset[self.dataset.columns[-1]].astype('category')
                self.dataset_y = self.dataset[self.dataset.columns[-1]]

                # If true, generate classifiers for every fold of the cross validation.
                # Store the classifiers in a the directory named after the fold index in the model directory
                self.x_train = []
                self.y_train = []
                self.x_test = []
                self.y_test = []
                
                self.trained_models = []
                self.train_time = []

                index = 0
                # Declare the stratified k fold object
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
                # Loop through the different folds
                for train_index, test_index in skf.split(self.dataset_x, self.dataset_y):
                    self.logger.info("[IoTDevID training] : Starting Fold number: {}".format(str(index)))
                    # split the dataset into train and test dataset using the indices
                    self.x_train.append(self.dataset_x.iloc[train_index])
                    self.y_train.append(self.dataset_y.iloc[train_index])
                    self.x_test.append(self.dataset_x.iloc[test_index])
                    self.y_test.append(self.dataset_y.iloc[test_index])
                    
                    m_train = self.x_train[index]["MAC"]
                    del self.x_train[index]["MAC"]

                    train_time, models = train_model(self.x_train[index], self.y_train[index], ml_algo)
                    self.trained_models.append(models)
                    self.train_time.append(train_time)

                    model_path = os.path.join(self.model_base_path, self.train_data["name"] + "_" + ml_algo + "_" + str(index) + ".sav")
                    pickle.dump(models, open(model_path, 'wb'))
                    
                    self.logger.info("[IoTDevID training] : Finished training the model for the fold number: {}".format(str(index)))
                    self.logger.info("[IoTDevID training] : Model is saved in the location: {}".format(model_path))

                    index += 1
            
            else:
                self.logger.debug("[IoTDevID training] : Start training with {} dataset".format(self.train_data["name"]))

                m_train = self.dataset["MAC"]
                del self.dataset["MAC"]

                self.dataset_x = self.dataset[self.dataset.columns[0:-1]]
                self.dataset_x = self.dataset_x.to_numpy()

                self.dataset[self.dataset.columns[-1]] = self.dataset[self.dataset.columns[-1]].astype('category')
                self.dataset_y = self.dataset[self.dataset.columns[-1]]

                model_path = os.path.join(self.model_base_path, self.train_data["name"] + "_" + ml_algo + ".sav")
                
                if os.path.exists(model_path):
                    self.trained_models = pickle.load(open(model_path, 'rb'))
                else:
                    self.train_time, self.trained_models = train_model(self.dataset_x, self.dataset_y, ml_algo)
                    pickle.dump(self.trained_models, open(model_path, 'wb'))


        except Exception as e:
            self.logger.exception("[IoTDevID training] : {}".format(e))
            return True
        

    def get_models(self):
        """
        """
        return self.trained_models
