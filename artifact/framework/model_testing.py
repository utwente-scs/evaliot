import datetime
import json
import logging
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

class ModelTesting:
    """
    Base class for model testing
    This class needs to be inherited by the model testing classes to test different methods.
    """

    def __init__ (self, config, trained_models, train_object, test_data):
        """
        Sets the local variables from the config values
        """
        # The logger object to log the testing process
        self.logger = logging.getLogger(__name__)
        # The config in dict format
        self.config = config
        # The config of the test data (with data loaded and preprocessed)
        self.test_data = test_data
        # Models trained in the model training phase
        self.trained_models = trained_models
        # The model training class object
        self.train_object = train_object

        ### The following lists need to populated in the run function
        # True values
        self.true_values = []
        # Predicted values
        self.pred_values = []
        # Probability values for the predictions
        self.prob_values = []
    
    def run(self):
        """
        This function needs to be implemented by the inheriting class.
        In this function the testing of the model will be performed.
        """
        pass

    def generate_result(self, filepath):
        """
        Generates the result of the model testing in a json file

        [Args]
        filepath: path of the json file
        """

        self.logger.debug("[Model Testing] : Generating the result of the model testing of the method: {}"
                          .format(self.config["general"]["method-name"])
                        )
        # Put the results in a dict
        result = {}
        # Add the general information about the evaluation tests
        result["method"] = self.config["general"]["method-name"]
        result["timestamp"] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        result["eval-config"] = self.config["data-preprocessing"]
        
        is_multilabel = False

        if self.train_object.cross_validation:
            # If the model has enabled cross-validation compute iteration-wise stats and overall stats
            self.logger.debug("[Model Testing] : The model has enabled cross-validation.")
            result["cross-validation"] = True

            all_true_values = []
            all_pred_values = []

            if len(self.pred_values) > 0 and len(self.pred_values[0]) > 0 and type(self.pred_values[0][0]) == list:
                is_multilabel = True
            for index in range(len(self.true_values)):
                if len(self.true_values[index]) == len(self.pred_values[index]):
                    all_true_values.extend(self.true_values[index])
                    all_pred_values.extend(self.pred_values[index])
                else:
                    self.logger.warning("[Model Testing] : The true values and predicted values are not of the same length. ")
                    # If the length is not the same, check if the probablity values contain all 3 values
                    # i.e. (true, predicted, probability)
                    if len(self.prob_values[index][0]) == 3:
                        true_values, pred_values, prob_values = zip(*self.prob_values[index])
                        all_true_values.extend(true_values)
                        all_pred_values.extend(pred_values)
                    else:
                        self.logger.error("[Model Testing] :  The probability values are not of the correct length\
                                          to include true, predicted and probability values.")
                        
                        result["true-values-{}".format(str(index))] = self.true_values[index]
                        result["predicted-values-{}".format(str(index))] = self.pred_values[index]
            
            if is_multilabel:
                # Generate a list of unique labels from all the true values
                all_labels = sorted(list(set(x[0] for x in all_true_values)))
                # Generate a multi-label binarizer to train the multi-label model
                mlb = MultiLabelBinarizer()
                true_values_mlb = mlb.fit_transform(all_true_values)
                pred_values_mlb = mlb.transform(all_pred_values)
                result["devices"] = all_labels
                result["full-classification-report"] = classification_report(
                                                                                true_values_mlb,
                                                                                pred_values_mlb,
                                                                                labels=np.arange(0,len(all_labels),1),
                                                                                target_names=all_labels,
                                                                                zero_division=0.0,
                                                                                output_dict=True
                                                                            )
            else:
                if type(all_true_values) is np.ndarray:
                    all_labels = sorted(np.unique(all_true_values).tolist())
                else:
                    all_labels = sorted(list(set(all_true_values)))
                result["devices"] = all_labels
                result["full-classification-report"] = classification_report(
                                                                                all_true_values,
                                                                                all_pred_values,
                                                                                labels=all_labels,
                                                                                zero_division=0.0,
                                                                                output_dict=True
                                                                            )

        else:
            self.logger.debug("[Model Testing] : The model did not have cross-validation.")
            result["cross-validation"] = False

            if len(self.pred_values) > 0 and type(self.pred_values[0]) == list:
                is_multilabel = True
            
            # Generate a list of unique labels from the true values
            if type(self.train_object.dataset_y) is np.ndarray:
                all_labels = sorted(np.unique(self.train_object.dataset_y).tolist())
            else:
                all_labels = sorted(list(set(self.train_object.dataset_y)))
            
            if not self.config["data-preprocessing"]["use-known-devices"]:
                for index in range(len(self.true_values)):
                    if type(self.true_values[index]) == list:
                        if self.true_values[index][0] not in all_labels:
                            self.true_values[index] = ["Unknown"]
                    else:
                        if self.true_values[index] not in all_labels:
                            self.true_values[index] = "Unknown"

                all_labels.append("Unknown")

            # Check if the length of true values and predicted values are the same
            if len(self.true_values) == len(self.pred_values):
                if is_multilabel:
                    mlb = MultiLabelBinarizer()
                    mlb_input = [ [label] for label in all_labels]
                    mlb.fit(mlb_input)
                    true_values_mlb = mlb.transform(self.true_values)
                    print(mlb.classes_, len(mlb.classes_))
                    pred_values_mlb = mlb.transform(self.pred_values)

                    result["devices"] = all_labels
                    result["classification-report"] = classification_report(
                                                                            true_values_mlb,
                                                                            pred_values_mlb,
                                                                            labels=np.arange(0,len(all_labels),1),
                                                                            target_names=all_labels,
                                                                            zero_division=0.0,
                                                                            output_dict=True
                                                                        )
                else:
                    result["devices"] = all_labels
                    result["classification-report"] = classification_report(
                                                                            self.true_values,
                                                                            self.pred_values,
                                                                            labels=all_labels,
                                                                            zero_division=0.0,
                                                                            output_dict=True
                                                                        )
            else:
                # If the length is not the same, check if the probablity values contain all 3 values
                # i.e. (true, predicted, probability)
                self.logger.warning("[Model Testing] : The true values and predicted values are not of the same length. ")
                if len(self.prob_values[0]) == 3:
                    true_values, pred_values, prob_values = zip(*self.prob_values)
               
                    if is_multilabel:
                        mlb = MultiLabelBinarizer()
                        true_values_mlb = mlb.fit_transform(true_values)
                        pred_values_mlb = mlb.transform(pred_values)
                        result["devices"] = all_labels
                        result["classification-report"] = classification_report(
                                                                                true_values_mlb,
                                                                                pred_values_mlb,
                                                                                labels=np.arange(0,len(all_labels),1),
                                                                                target_names=all_labels,
                                                                                zero_division=0.0,
                                                                                output_dict=True
                                                                            )
                    else:
                        result["devices"] = all_labels
                        result["classification-report"] = classification_report(
                                                                                true_values,
                                                                                pred_values,
                                                                                labels=all_labels,
                                                                                zero_division=0.0,
                                                                                output_dict=True
                                                                            )
                else:
                    self.logger.warning("[Model Testing] :  The probability values are not of the correct length\
                                     to include true, predicted and probability values. ")
                    result["true-values"] = self.true_values
                    result["predicted-values"] = self.pred_values

        result["predictions"] = self.prob_values

        report = open(filepath, "w")
        report.write(json.dumps(result, indent=2))
        report.close()
