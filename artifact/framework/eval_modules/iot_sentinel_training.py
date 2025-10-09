import os
import pickle
import sys

from sklearn.model_selection import StratifiedKFold

from framework.model_training import ModelTraining

class IoTSentinelTraining(ModelTraining):

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

            # Append the path to the IoT sentinel repository to the system paths
            # This allows for importing functions and classes from the repository like a library
            sys.path.append(self.paths["repo"])
            from IoTSentinel import load_data

            self.logger.debug("[IoT Sentinel training] : Imported IoT Sentinel code from the repo")

            # Define the path of the dir to store the processed data as pickle files
            pickle_data_dir = os.path.join(self.paths["eval-dir"], train_data["name"])
            # Generate the dir if it does not exist
            if not os.path.exists(pickle_data_dir):
                os.makedirs(pickle_data_dir)
            
            # Process the pcap files and get the corresponding dataset and distance vector
            self.dataset_x, self.dataset_y, self.vectors_edit_distance = \
                                                load_data(train_data["file-list"], pickle_data_dir)
            self.logger.debug("[IoT Sentinel training] : Loaded the training dataset")
        
        except Exception as e:
            self.logger.exception("[IoT Sentinel training] : {}".format(e))
            self.logger.error("[IoT Sentinel training] : ERROR importing the IoT Sentinel code from the repo or loading the training dataset")


    def run(self):
        """
        Runs the code train the model using the IoT Sentinel code. 
        """

        try:
            from IoTSentinel import train_model

            same_to_other_ratio = 10
            self.logger.debug("[IoT Sentinel training] : Training the model using the IoT Sentinel code")
            
            if self.cross_validation:
                # If true, generate classifiers for every fold of the cross validation.
                # Store the classifiers in a the directory named after the fold index in the model directory
                self.x_train = []
                self.y_train = []
                self.x_test = []
                self.y_test = []
                # List of classifiers for each fold are saved as a list on this list i.e [{fold1}, {fold2}...]
                self.classifier_list = []

                # Start the Straitified k fold validation here with 5 folds                
                index = 0
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
                for train_index, test_index in skf.split(self.dataset_x, self.dataset_y):
                    self.logger.debug("[IoT Sentinel training] : Starting fold number {} for cross validation training."
                                      .format(str(index)))
                    self.x_train.append(self.dataset_x[train_index])
                    self.y_train.append(self.dataset_y[train_index])
                    self.x_test.append(self.dataset_x[test_index])
                    self.y_test.append(self.dataset_y[test_index])

                    self.logger.debug("[IoT Sentinel training] : Number of packets for training fold {} is : {}"
                                          .format(str(index), str(len(self.y_train[index]))))

                    # Get the number of packets per device used in for the training.
                    label_counts = {}
                    for y in self.y_train[index]:
                        if y not in label_counts:
                            label_counts[y] = 0
                        label_counts[y] += 1

                    self.logger.debug("[IoT Sentinel training] : Number of packets per device for training fold {} is : {}"
                                          .format(str(index), label_counts))
                    
                    self.classifier_list.append( 
                                                    train_model( 
                                                        self.x_train[index],
                                                        self.y_train[index],
                                                        same_to_other_ratio,
                                                        os.path.join(self.paths["model-dir"], str(index))
                                                    )
                                                )
                    index += 1

                self.logger.debug("[IoT Sentinel training] : Generated {} set of classifiers for cross validation".format(index))
            
            else:
                # Get the number of packets per device used in for the training.
                label_counts = {}
                for y in self.dataset_y:
                    if y not in label_counts:
                        label_counts[y] = 0
                    label_counts[y] += 1

                self.logger.debug("[IoT Sentinel training] : Total number of packets used for training is : {}"
                                          .format(len(self.dataset_y)))
                self.logger.debug("[IoT Sentinel training] : Number of packets per device for training is : {}"
                                          .format(label_counts))
                
                # If false, generate a single classifier per device using all the training data
                self.classifier_list = train_model(
                                                self.dataset_x,
                                                self.dataset_y,
                                                same_to_other_ratio,
                                                self.paths["model-dir"]
                                            ) 
                self.logger.debug("[IoT Sentinel training] : Generated a classifiers for the dataset: {}!".format(self.train_data["name"]))
            return False

        except Exception as e:
            self.logger.exception("[IoT Sentinel training] : {}".format(e))
            return True


    def get_models(self):
        """
        if the classifier_list exists, then return it. Otherwise, load the models
        """
        
        try:
            self.logger.debug("[IoT Sentinel training] : Returning the classifier models from the memory")
            # If the variable classifier_list hasn't been declared yet, it will throw an error
            return self.classifier_list
        except:
            self.logger.debug("[IoT Sentinel training] : Loading the classifier models saved as files")
            # Load the models from the path
            self.classifier_list = {}

            for model_file in os.listdir(self.paths["model-dir"]):
                if model_file.endswith(".sav"):
                    # get the device name from the model path
                    # model files are named as "model_[device-name].sav"
                    device_name = os.path.basename(model_file).split(".")[0].split("_")[1]
                    model_path = os.path.join(self.paths["model-dir"], model_file)
                    clf = pickle.load(open(model_path, 'rb'))
                    self.classifier_list[device_name] = clf
            
            self.logger.debug("[IoT Sentinel training] : Returning the loaded classifier models")
            return self.classifier_list()