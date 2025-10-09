import logging
from scapy.all import *

class ModelTraining:
    """
    Base class for model training.
    This class needs to be inherited by the model training classes to train different methods.
    """
    
    def __init__(self, config, train_data, cross_validation):
        """
        Sets the local variables from the config values

        config: config values in dict form
        """
        # The logger object to log the training process
        self.logger = logging.getLogger(__name__)
        # The paths listed in the config of model-training
        self.paths = config["paths"]
        # The config of the train data (with data loaded and preprocessed)
        self.train_data = train_data
        # The cross validation flag
        self.cross_validation = cross_validation

    def run(self):
        """
        This function needs to be implemented by the inheriting class.
        Runs the command to train the model.
        """
        pass

    def get_models(self):
        """
        Read the models stored in the directory, and return them
        """
        pass

