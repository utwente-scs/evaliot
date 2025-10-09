import os
import sys

from framework.model_testing import ModelTesting


class MudgeeTesting(ModelTesting):


    def __init__(self, config, trained_models, train_object, test_data):
        """
        Initialize a mudgee testing class object with the following
        arguments:

        [Args]
        config: 
        model_dir:
        train_object:
        """
        super().__init__(config, trained_models, train_object, test_data)

        try:
            sys.path.append(config["model-training"]["paths"]["repo"])
            self.logger.debug("[MUDgee testing] : Imported MUDgee code from the repo")
        except Exception as e:
            self.logger.exception("[MUDgee testing] : {}".format(e))
            self.logger.error("[MUDgee testing] : ERROR importing the MUDgee code from the repo")

    
    def run(self):
        """
        Run the Mudgee Tests
        """

        try:
            from src.profile_handling import runtime_profile_generation

            cfg = {}
            cfg["default-gateway-ip"] = self.config["model-testing"]["default-gateway-ip"]
            
            self.logger.debug("[MUDgee testing] : Testing the MUDGEE code")
            # If cross validation is true 
            if self.train_object.cross_validation:

                if len(self.trained_models) != self.train_object.num_folds:
                    self.logger.error("[MUDgee testing] :  The set of MUD profiles loaded is not equal to the number of fold for cross validations!")
                    exit(1)

                # Generate the temp directory
                temp_dir = os.path.join(self.config["model-training"]["paths"]["eval-dir"], "temp")
                os.makedirs(temp_dir, exist_ok=True)

                self.true_values, self.pred_values, self.prob_values = [], [], []
                for index in range(self.train_object.num_folds):
                    self.logger.debug("[MUDgee testing] : Starting fold number {} of cross validation.".format(str(index)))
                    true_values, pred_values, prob_values = [], [], []
                    for device in self.test_data["device-names"]:
                                               
                        # Move the file used for training to temp directory so it is not used in the testing
                        train_file_path = self.train_object.selected_all[device][index]
                        new_file_path = os.path.join(temp_dir, os.path.basename(train_file_path))
                        os.rename(train_file_path, new_file_path)

                        cfg["device-name"] = device
                        cfg["dir-pcaps"] = self.test_data["path"] + "/" + device
                        
                        dynamic_matches, static_matches = runtime_profile_generation(cfg, self.trained_models[index])
                        self.logger.debug("[MUDgee testing] : Retrieved the similarity values for the device : {}.".format(device))
                        
                        true, pred, prob = self.get_similarity_matches(device, dynamic_matches, static_matches)
                        true_values.extend(true)
                        pred_values.extend(pred)
                        prob_values.extend(prob)

                        self.logger.debug("[MUDgee testing] : The device: {} was predicted to be: {}".format(device, pred))

                        # Move the file back to the original directory
                        os.rename(new_file_path, train_file_path)

                    
                    self.true_values.append(true_values)
                    self.pred_values.append(pred_values)
                    self.prob_values.append(prob_values)
                
                self.logger.debug("[MUDgee testing] : MUDgee tested with cross validation for the dataset: {}!".format(self.train_object.train_data["name"]))
            
            else:
                for device in self.test_data["device-names"]:
                    cfg["device-name"] = device
                    cfg["dir-pcaps"] = self.test_data["path"] + "/" + device
                    dynamic_matches, static_matches = runtime_profile_generation(cfg, self.trained_models)

                    true, pred, prob = self.get_similarity_matches(device, dynamic_matches, static_matches)
                    self.true_values.extend(true)
                    self.pred_values.extend(pred)
                    self.prob_values.extend(prob)
                                        
                    self.logger.debug("[MUDgee testing] : The device: {} was predicted to be: {}".format(device, pred))
    
                self.logger.debug("[MUDgee testing] : MUDgee tested for the dataset: {}!".format(self.test_data["name"]))

        except Exception as e:
            self.logger.exception("[MUDgee testing] : {}".format(e))
            self.logger.debug("[MUDgee testing] : ERROR running the MUDGEE code")

        
    def get_similarity_matches(self, device, dynamic_matches, static_matches):
        """
        Select the winner for every epoch iteration between the dynamic and static matches.
        """
        true_values = []
        pred_values = []
        prob_values = []

        for index in range(len(dynamic_matches)):
            d_score, d_devices = list(dynamic_matches[index].items())[0]
            s_score, s_devices = list(static_matches[index].items())[0]

            if d_score == 1:
                true_values.append([device])
                pred_values.append(d_devices)
                prob_values.append(([device], d_devices, (d_score, s_score)))
            elif s_score == 1:
                true_values.append([device])
                pred_values.append(s_devices)
                prob_values.append(([device], s_devices, (d_score, s_score)))
            elif len(d_devices) == 1 and len(s_devices) == 1 and d_devices[0] == s_devices[0]:
                true_values.append([device])
                pred_values.append(d_devices)
                prob_values.append(([device], d_devices, (d_score, s_score)))
            elif d_score >= self.config["data-preprocessing"]["threshold"][0]:
                true_values.append([device])
                pred_values.append(d_devices)
                prob_values.append(([device], d_devices, (d_score, s_score)))
            elif s_score >= self.config["data-preprocessing"]["threshold"][1]:
                true_values.append([device])
                pred_values.append(s_devices)
                prob_values.append(([device], s_devices, (d_score, s_score)))
            else:
                true_values.append([device])
                if not self.config["data-preprocessing"]["use-known-devices"]:
                    pred = "Unknown"
                else:
                    pred = ""
                pred_values.append([pred])
                prob_values.append(([device], [pred], (d_score, s_score)))

        return true_values, pred_values, prob_values            




