import collections
import os
import sys
import traceback

from framework.model_testing import ModelTesting

class IoTSentinelTesting(ModelTesting):


    def __init__(self, config, trained_models, train_object, test_data):
        """
        Sets the local variables from the config values

        config: config values in dict form
        """
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            from IoTSentinel import load_data

            self.logger.debug("[IoT Sentinel testing] : Imported IoT Sentinel code from the repo")

            pickle_data_dir = os.path.join(config["model-training"]["paths"]["eval-dir"], test_data["name"])
            if not os.path.exists(pickle_data_dir):
                os.makedirs(pickle_data_dir)

            self.dataset_x, self.dataset_y, self.vectors_edit_distance = load_data(
                                                                                    self.test_data["file-list"],
                                                                                        pickle_data_dir
                                                                                    )
            
            self.logger.debug("[IoT Sentinel testing] : Loaded the testing dataset")

        except Exception as e:
            self.logger.exception("[IoT Sentinel testing] : {}".format(e))
            self.logger.error("[IoT Sentinel testing] : ERROR importing the IoT Sentinel code from the repo or loading the test dataset")
        

    def run(self):
        """
        Runs the command to test the model using the IoT Sentinel code
        """
        try:
            from IoTSentinel import calc_feature_importance, test_model

            self.dev_pred_accuracy = {}
            self.f_importance = {}
            self.iterationwise_fimportance = {}
            self.test_dev_counter = {}
            self.number_of_features = 23
            
            self.logger.debug("[IoT Sentinel testing] : Testing the IoT Sentinel code")
            try:
                # Get the train and test data here using in index
                if self.train_object.cross_validation:

                    for index in range(len(self.train_object.x_test)):
                        self.logger.info("[IoT Sentinel testing] : Starting fold number {} of cross validation testing."
                                         .format(str(index)))
                        self.logger.debug("[IoT Sentinel testing] : Number of packets for testing fold {} is : {}"
                                          .format(str(index), str(len(self.train_object.y_test[index]))))

                        curr_test_dev_counter = collections.Counter(self.train_object.y_test[index])
                        self.test_dev_counter = { k: self.test_dev_counter.get(k, 0) + curr_test_dev_counter.get(k, 0)
                                                for k in set(self.test_dev_counter) | set(curr_test_dev_counter) }
                        
                        for device in self.trained_models[index]:
                            self.f_importance, self.iterationwise_fimportance = \
                                        calc_feature_importance(
                                                            self.trained_models[index][device],
                                                            self.number_of_features,
                                                            self.f_importance,
                                                            self.iterationwise_fimportance
                                                        )

                        true_values, pred_values, prob_values = \
                                        test_model(self.train_object.x_test[index], self.train_object.y_test[index],
                                                    self.train_object.x_train[index], self.train_object.y_train[index],
                                                    self.trained_models[index], self.vectors_edit_distance,
                                                    self.dev_pred_accuracy)
                        self.true_values.append(true_values)
                        self.pred_values.append(pred_values)
                        self.prob_values.append(prob_values)
                    
                    self.logger.debug("[IoT Sentinel testing] : Testing complete with cross validation")
                else:
                    curr_test_dev_counter = collections.Counter(self.dataset_y)
                    self.test_dev_counter = { k: self.test_dev_counter.get(k, 0) + curr_test_dev_counter.get(k, 0)
                                            for k in set(self.test_dev_counter) | set(curr_test_dev_counter) }
                    
                    for device in self.trained_models:
                        self.f_importance, self.iterationwise_fimportance = \
                                    calc_feature_importance(
                                                        self.trained_models[device],
                                                        self.number_of_features,
                                                        self.f_importance,
                                                        self.iterationwise_fimportance
                                                    )

                    self.true_values, self.pred_values, self.prob_values = \
                                    test_model(self.dataset_x, self.dataset_y,
                                               self.train_object.dataset_x, self.train_object.dataset_y,
                                               self.trained_models, self.vectors_edit_distance,
                                               self.dev_pred_accuracy)
                    
                    if not self.config["data-preprocessing"]["use-known-devices"]:
                        print("Check for unknown devices", len(self.true_values), len(self.pred_values), len(self.prob_values))
                        for i in range(0, len(self.true_values)):
                            print(i, self.prob_values[i])
                            if self.prob_values[i][2] < self.config["data-preprocessing"]["threshold"]:
                                self.pred_values[i] = "Unknown"
                                temp_val = list(self.prob_values[i])
                                temp_val[1] = "Unknown"
                                self.prob_values[i] = tuple(temp_val)
                                print(self.prob_values[i])
                    
                    
                    self.logger.debug("[IoT Sentinel testing] : Testing complete without cross validation")

            except Exception as e:
                self.logger.exception("[IoT Sentinel testing] : ".format(e))
                self.logger.error("[IoT Sentinel testing] : ERROR! Unable to get the train and test data from the train object")

            return False
        except Exception as e:
            self.logger.exception("[IoT Sentinel testing] : ".format(e))
            return True
    