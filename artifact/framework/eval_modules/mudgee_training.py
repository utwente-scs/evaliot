import json
import os
import random
import sys

from framework.model_training import ModelTraining


class MudgeeTraining(ModelTraining):

    def __init__(self, config, train_data, cross_validation):
        """
        Sets the local variables from the config values

        config: config values in dict form
        """
        super().__init__(config["model-training"], train_data, cross_validation)
        self.merge_cmd = config["model-training"]["cmds"]["merge"]
        self.update_cmd = config["model-training"]["cmds"]["update"]
        self.train_cmd = config["model-training"]["cmds"]["train"]
        self.dataset_y = list(self.train_data["file-list"].keys())

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[MUDgee training] : Imported MUDgee code from the repo")
        except Exception as e:
            self.logger.exception("[MUDgee training] : {}".format(e))
            self.logger.error("[MUDgee training] : ERROR importing the MUDgee code from the repo")
        

    def run(self):
        """
        Runs the command to train the model using the MUDGEE code
        """
        # store the current working directy before changing it
        # to the home directiry of the method being tested
        pwd = os.getcwd()
        os.chdir(self.paths["eval-dir"])
        
        self.logger.debug("[MUDgee training] : Training the model using the MUDGEE code")
        # If training and testing dataset are the same, we need to perform cross validation
        if self.cross_validation:
            # Initialize this to the number of folds
            self.num_folds = 5
            # This will store selected files for each device per fold
            self.selected_all = {}
            # Run the MUD profile generation for every fold
            for index in range(self.num_folds):
                # Files selected in this fold.
                selected = {}
                for device in self.train_data["file-list"]:
                    if device not in self.selected_all:
                        self.selected_all[device] = []
                    iter = 0
                    while True:
                        # Select a file at random from the file list for the device
                        selected_file = random.choice(self.train_data["file-list"][device])
                        iter += 1
                        # If the file wasn't selected before in a previous fold, then add it to the list
                        if selected_file not in self.selected_all[device]:
                            self.selected_all[device].append(selected_file)
                            break
                        elif iter >= 5:
                            self.selected_all[device].append(selected_file)
                            break
                        # Else continue and find a new file at random
                        else:
                            continue
                    # Add the selected file to the selected dictionary
                    selected[device] = selected_file
                # Run the update script to generate the MUDGEE config file for the selected files
                res = os.system(self.update_cmd.format(self.train_data["name"], json.dumps(selected), str(index)))
                # If there was an error, break out of the loop
                if res==0:
                    self.logger.debug("[MUDgee Training] : Configs updated for training the MUD profile in fold {}".format(str(index)))
                else:
                    self.logger.error("[MUDgee Training] : ERROR updating the MUDGEE config file in fold {}".format(str(index)))
                    return res
                # Run the MUDGEE script to generate the MUD profiles
                res = os.system(self.train_cmd.format(self.train_data["name"]) + " " + str(index))
                # If there was an error, break out of the loop
                if res==0:
                    self.logger.debug("[MUDgee Training] : MUD profiles generared in the fold {}".format(str(index)))
                else:
                    self.logger.error("[MUDgee Training] : ERROR generating MUD profiles in fold {}".format(str(index)))
                    return res
        else:
            # Run the merge script to merge pcaps of a single device into a single file
            merge_dir = self.train_data["path"].rstrip("/")+"-merged"
            merge_file_map = {}
            if not os.path.exists(merge_dir):
                os.makedirs(merge_dir)

            for device in self.train_data["file-list"]:
                merged_file_path = os.path.join(merge_dir, device+".pcap")
                merge_file_map[device] = merged_file_path
                if os.path.exists(merged_file_path):
                    continue
                device_path = os.path.join(self.train_data["path"], device)
                res = os.system(self.merge_cmd.format(merged_file_path, device_path))
                if res == 0:
                    self.logger.debug("[MUDgee Training] : Merged pcaps for device {} into {}".format(device, merged_file_path))
                else:
                    self.logger.error("[MUDgee Training] : ERROR merging pcaps for device {} into {}".format(device, merged_file_path))
                    continue
            
            # Run the update script to update the MUDGEE config files with the merged files
            res = os.system(self.update_cmd.format(self.train_data["name"], json.dumps(merge_file_map), "-"))
            if res==0:
                self.logger.debug("[MUDgee Training] : Configs updated for training the MUD profile!")
            else:
                self.logger.error("[MUDgee Training] : ERROR updating the MUDGEE config file!")
                return res
            # Run the MUDGEE script to generate the MUD profiles
            res = os.system(self.train_cmd.format(self.train_data["name"]))
            if res==0:
                self.logger.debug("[MUDgee Training] : MUD profiles generared for the dataset: {}!".format(self.train_data["name"]))
            else:
                self.logger.error("[MUDgee Training] : ERROR generating MUD profiles for the dataset: {}!".format(self.train_data["name"]))
                return res
        
        # change directory back to the original directory
        os.chdir(pwd)
        
        return res
    

    def get_models(self):
        """
        Retreives the MUDGEE profiles from the directory where they are stored
        """
        try: 
            from src.profile_handling import load_mud_profiles
            if self.cross_validation:
                mud_profiles = []
                for iter in range(self.num_folds):
                    model_dir = os.path.join(self.paths["model-dir"].format(self.train_data["name"]), str(iter))
                    if os.path.isdir(model_dir):
                        mud_profiles.append(load_mud_profiles(model_dir))
                    else:
                        self.logger.error("[MUDgee Training] : ERROR! The path for MUD profiles doesn't exist at path: {}".format(model_dir))
                        return None
            else:
                model_dir = os.path.join(self.paths["model-dir"].format(self.train_data["name"]), "result")
                if os.path.isdir(model_dir):
                    mud_profiles = load_mud_profiles(model_dir)
                else:
                    self.logger.error("[MUDgee Training] : ERROR! The path for MUD profiles doesn't exist at path: {}".format(model_dir))
                    return None
            
            return mud_profiles
                
        except Exception as e:
            self.logger.exception("[MUDgee Training] : {}".format(e))
            self.logger.error("[MUDgee Training] : ERROR retreiving the MUDGEE profiles")
            return None