
from datetime import datetime
import importlib
import logging
import os
import sys
import yaml

from framework.data_ingestion import DataIngestion
from framework.data_preprocessing import DataPreprocessing
from framework.utils import verify_config


def import_class(class_name, class_path):
    """
    Imports the class from the class_path with the class_name
    Used to import evaluation module classes

    [Args]
    class_name: name of the class
    class_path: path to the class file
    """
    spec = importlib.util.spec_from_file_location(class_name, class_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)


def main_controller(logger, cfg):
    """
    Main controller function for the evaluation

    [Args]
    logger: logger object
    cfg: config object
    """
    ###########################################################
    #                    Data Ingestion                      #
    ###########################################################

    try:
        # Create the object for the data ingestion class using the config values.
        data_ingestion = DataIngestion(cfg["data-ingestion"], cfg["general"])
        
        input_datasets = data_ingestion.list_datasets

        logger.debug( "[Data Ingestion] : Datasets Ingested: {}".format(len(input_datasets)))
        # print(input_datasets)

    except Exception as e:
        # Catch the exception and log it
        logger.exception("[Data Ingestion] : {}".format(e))
    except KeyboardInterrupt:
        # Catch the keyboard interrupt and log it
        logger.error("[Data Ingestion] Keyboard Interrupt")
        sys.exit()


    ###########################################################
    #                     Data Preprocessing                  #
    ###########################################################
    
    try:

        data_preprocess = DataPreprocessing(
                                            cfg["data-preprocessing"],
                                            cfg["general"]["input-type"],
                                            input_datasets
                                        )

        # if data format required is not "raw
        # if cfg["data-preprocessing"]["required-data-format"] != "raw":
        #     data_preprocess.prepocess_input()

        # Split the input files for testing and training and
        # read the data from them.
        train_data, test_data = data_preprocess.get_dataset_info()
        if "device-names" in train_data and "device-names" in test_data:
            logger.debug( "[Data Prepocessing] : Training data loaded with devices: %s, Testing data loaded with devices: %s", train_data["device-names"], test_data["device-names"])
    
    except Exception as e:
        # Catch the exception and log it
        logger.exception("[Data Prepocessing] : {}".format(e))
        exit(0)

    except KeyboardInterrupt:
        # Catch the keyboard interrupt and log it
        logger.error("[Data Prepocessing] : Keyboard Interrupt")
        sys.exit()

    ###########################################################
    #                   Load Training Class                   #
    ###########################################################
    
    try:
        import_class(cfg["model-training"]["class-name"], cfg["model-training"]["class-path"])
        
        # Get the model training class from the config
        model_training_class = getattr(
                                        sys.modules[cfg["model-training"]["class-name"]],
                                        cfg["model-training"]["class-name"]
                                    )
        logger.debug("Generated Model Training Class:{}".format( cfg["model-training"]["class-name"]))
        logger.debug(model_training_class)
        
    except Exception as e:
        # Catch the exception and log it
        logger.error("[Load Model Training Class] : Error loading the model training class. ")
        logger.exception("[Load Model Training Class] : {}".format(e))
        exit(0)


    ###########################################################
    #                    Load Testing Class                   #
    ###########################################################

    try: 
        import_class(cfg["model-testing"]["class-name"], cfg["model-testing"]["class-path"])

        # Get the model testing class from the config
        model_testing_class = getattr(
                                        sys.modules[cfg["model-testing"]["class-name"]],
                                        cfg["model-testing"]["class-name"]
                                        )
        logger.debug("Generated Model Test Class: {}".format(cfg["model-testing"]["class-name"]))
        logger.debug(model_testing_class)
    
    except Exception as e:
        # Catch the exception and log it
        logger.error("[Load Model Testing Class] : Error loading the model testing class. ")
        logger.exception("[Load Model Testing Class] : {}".format(e))
        exit(0)

    ###########################################################
    #                    Cross Validation                     #
    ###########################################################

    # If the training and testing dataset is the same, we need to perform cross validation
    cross_validation = cfg["data-preprocessing"]["train-dataset"] == cfg["data-preprocessing"]["test-dataset"]
    
    # Update the logger to include the cross validation value
    logger.debug("Training Dataset: {}".format(cfg["data-preprocessing"]["train-dataset"]))
    logger.debug("Testing Dataset: {}".format(cfg["data-preprocessing"]["test-dataset"]))
    logger.debug("Cross Validation: {}".format(cross_validation))

   
    ###########################################################
    #                    Model Training                       #
    ###########################################################

    try:
        # Initiate the class with the config values
        model_trainer = model_training_class(
                                                cfg,
                                                train_data,
                                                cross_validation
                                            )

        # check if the model needs to be trained.
        if cross_validation or cfg["model-training"]["train-model"]:
            # processed_training_data = data_preprocess.prepocess_data(training_data)
            if model_trainer:
                logger.debug("Model Train Class Initialized! Running the command to train the model...")
                # Run the model trainer
                res = model_trainer.run()
                # Log the result
                logger.debug("[Model Training]: Result: {}".format(res))
                # If there was an error, log it
                if res:
                    logger.error("[Model Training] : ERROR! Unable to run the command to train the model!")
                    exit(0)
            else:
                logger.error("[Model Training] : ERROR! Model trainer is not initialized!")
                exit(0)
        
        # get stored models from their storage directory
        trained_models = model_trainer.get_models()
    
    except Exception as e:
        logger.exception("[Model Training] : {}".format(e))
        exit(0)



    ###########################################################
    #                     Model Testing                       #
    ###########################################################
    try:
        # # Preprocess the test data
        if test_data["load"] == True:
            processed_test_data = data_preprocess.preprocess_data(test_data)
            logger.debug("[Model Testing] : Processed and loaded the test data")
        else:
            processed_test_data = test_data
            logger.debug("[Model Testing] : No processing or loading required for the test data")

        model_tester = model_testing_class(
                                    cfg,
                                    trained_models,
                                    model_trainer,
                                    processed_test_data
                                )
        if model_tester:
            logger.debug("[Model Testing] : Model Test Class Initialized! Running the command to test the model...")
            # Run the model tester
            model_tester.run()
        
        else:
            logger.error("[Model Testing] : ERROR! Model tester is not initialized!")
            exit(0)
                            
        # Verify if the dir exists
        if not os.path.exists(cfg["model-testing"]["report-dir"]):
            os.makedirs(cfg["model-testing"]["report-dir"])
        
        # Get the report name. Check if the report already exists, if so, generate a new name
        index = 0
        while True:
            # Generate the name for the report
            report_name = cfg["general"]["method-name"].lower() + "_" + test_data["name"] + "_" + str(index) + ".json"
            # Genetrate the path of the report file
            report_path = os.path.join(cfg["model-testing"]["report-dir"], report_name)
            if not os.path.exists(report_path):
                break
            else:
                index += 1
                continue
  
        # Call generate result to generate the report from the evaluations
        model_tester.generate_result(report_path)
        if os.path.exists(report_path):
            logger.debug("[Model Testing] : Result report generated at path: {}".format(report_path))        
        else:
            logger.error("[Model Testing] : ERROR! Result report NOT generated at path: {}".format(report_path))
    
    except Exception as e:
        logger.exception("[Model Testing] : {}".format(e))
        exit(0)




########################### Main Function ###########################


if __name__ == '__main__':

    ###########################################################
    #            Read and Verify the config file              #
    ###########################################################

    config_file = sys.argv[1]

    with open(config_file, 'r') as cfgfile:
        cfg = yaml.load(cfgfile, Loader=yaml.Loader)
        print(cfg)
    
    error = verify_config(cfg)

    if error:
        exit(0)
    
    ###########################################################
    #                    Set up the logger                    #
    ###########################################################
        
    if not os.path.exists(cfg["general"]["log-dir"]):
        os.makedirs(cfg["general"]["log-dir"])

    log_filename = cfg["general"]["method-name"].lower() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_filepath = os.path.join(cfg["general"]["log-dir"], log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.getLogger('matplotlib').disabled = True
    
    handler = logging.FileHandler(log_filepath)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    try:
        # Call main controller function to run the evaluation tests
        main_controller(logger, cfg)
    except Exception as e:
        # Catch the exception and log it
        logger.exception("{}".format(e))
        exit(0)
    except KeyboardInterrupt:
        # Catch the keyboard interrupt and log it
        logger.error("Keyboard Interrupt")
        sys.exit()