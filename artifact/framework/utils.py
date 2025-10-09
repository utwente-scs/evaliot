import csv
import json
import logging
import os

# Get the root logger
logger = logging.getLogger(__name__)


def verify_config(cfg):
    """
    Verify that the config values are correct

    [Args]
    cfg: Config file
    
    """
    logger.debug("[Config] : Verifying config file...")
    error = False
    # Check if "general" field is present
    if not cfg.get("general"):
        error = True
        raise ValueError("General config not found")
    
    # Check if "data-ingestion" field is present
    if not cfg.get("data-ingestion"):
        error = True
        raise ValueError("Data Ingestion config not found")
    else:
        # Check if "list-datasets" field is present
        if not cfg["data-ingestion"].get("list-datasets"):
            error = True
            raise ValueError("Data Ingestion list-datasets config not found")
        else:
            # Check if "list-datasets" is a list
            if not isinstance(cfg["data-ingestion"]["list-datasets"], list):
                error = True
                raise ValueError("Data Ingestion list-datasets config is not a list")
            else:
                # Check if "list-datasets" contains valid paths
                for dataset in cfg["data-ingestion"]["list-datasets"]:
                    if not os.path.exists(dataset["path"]):
                        error = True
                        raise ValueError("Data Ingestion dataset path {} doesn't exist".format(dataset["path"]))
                    
                    has_subdirs = any(os.path.isdir(os.path.join(dataset["path"], item)) for item in os.listdir(dataset["path"]))
                    
                    # Check if the dataset if configured correctly as per_device
                    if dataset["type"] == "per_device" and not has_subdirs:
                        error = True
                        raise ValueError("Dataset {} is INCORRECTLY configured as per_device. It does not have any subfolders.".format(dataset["name"]))

                    # Check if the "input-type" requirement for the method matches the type of the dataset.
                    if cfg["general"]["input-type"] == "per_device" and cfg["general"]["input-type"] != dataset["type"]:
                        error = True
                        raise ValueError("Input type: {} and Dataset Type: {} does not match!".format(cfg["general"]["input-type"], dataset["type"]))



    # Check if "data-preprocessing" field is present
    if not cfg.get("data-preprocessing"):
        error = True
        raise ValueError("Data Preprocessing config not found")
    else:
        datasets = []
        # Check if "train-dataset" and "test-dataset" are present in "data-ingestion" config
        for dataset in cfg["data-ingestion"]["list-datasets"]:
            datasets.append(dataset["name"])
        if cfg["data-preprocessing"].get("train-dataset") not in datasets:
            error = True
            raise ValueError("Data Preprocessing train-dataset {} doesn't exist in Data Ingestion".format(cfg["data-preprocessing"]["train-dataset"]))
        if cfg["data-preprocessing"].get("test-dataset") not in datasets:
            error = True
            raise ValueError("Data Preprocessing test-dataset {} doesn't exist in Data Ingestion".format(cfg["data-preprocessing"]["test-dataset"]))
            
    
    # Check if "model-training" fields are present
    if not cfg.get("model-training"):
        error = True
        raise ValueError("Model Training config not found")
    else:
        # Check if "class-name" field is present
        if not cfg["model-training"].get("class-name"):
            error = True
            raise ValueError("Model Training class-name config not found")

        # Check if "class-path" field is present
        if not cfg["model-training"].get("class-path"):
            error = True
            raise ValueError("Model Training class-path config not found")
        else:
            # Check if the path in "class-path" exists
            if not os.path.exists(cfg["model-training"]["class-path"]):
                error = True
                raise ValueError("Model Training class-path {} doesn't exist".format(cfg["model-training"]["class-path"]))
        # Check if "paths" field is present
        if not cfg["model-training"].get("paths"):
            error = True
            raise ValueError("Model Training paths config not found")
        else:
            # Check if the paths listed in "paths" exist
            for path in cfg["model-training"]["paths"]:
                if  not "model-dir" and not os.path.exists(cfg["model-training"]["paths"][path]):
                    error = True
                    raise ValueError("Model Training path {}:{} doesn't exist".format(path, cfg["model-training"]["paths"][path]))
        
        if not cfg["model-training"].get("train-model"):
            model_dir = cfg["model-training"]["paths"]["model-dir"]
            if "{}" in cfg["model-training"]["paths"]["model-dir"]:
                model_dir = cfg["model-training"]["paths"]["model-dir"].format(cfg["data-preprocessing"]["train-dataset"])
            if not os.path.exists(model_dir):
                error = True
                raise ValueError("Model Training model-dir {} doesn't exist. Change config to train a new model".format(cfg["model-training"]["paths"]["model-dir"]))
    
    # Check if "model-testing" fields are present
    if not cfg.get("model-testing"):
        error = True
        raise ValueError("Model Testing config not found")
    else:
        # Check if "class-name" field is present
        if not cfg["model-testing"].get("class-name"):
            error = True
            raise ValueError("Model Testing class-name config not found")
        
        # Check if "class-path" field is present
        if not cfg["model-testing"].get("class-path"):
            error = True
            raise ValueError("Model Testing class-path config not found")
        else:
            # Check if the path in "class-path" exists
            if not os.path.exists(cfg["model-testing"]["class-path"]):
                error = True
                raise ValueError("Model Testing class-path {} doesn't exist".format(cfg["model-testing"]["class-path"]))

    if error:
        # If any errors were raised, raise a ValueError
        logger.error("[Config] : Error loading the config file. Please check the config file and try again.")
        raise ValueError("One or more config values are missing")

    return error
