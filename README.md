# emberGAN

Using <a href="https://github.com/yanminglai/Malware-GAN"> Malware-GAN </a>, which is the realization of paper: <a href="https://arxiv.org/abs/1702.05983"> "Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN", 2017 </a>, we treat EmberNet as the blackbox model to be attacked.

# Setting up
<b>\__init__.py</b>: Replace EmberNet's \__init__.py with this (remember to run setup.py to set EmberNet up again). Added predict function that takes in jsonl file. 

<b>MalGAN_ember.py</b>: Trains MalGAN to attack EmberNet as a blackbox, then retrains EmberNet with the generated adversarial samples. Outputs result of training into compiled_results.csv. 

Change the following filepaths to the appropriate directories: </br>
-self.blackbox_modelpath (Path to EmberNet's model.h5 file) </br>
-self.ember_filepath (Path to test_features.jsonl provided by the EMBER dataset) 

<b>Info about helper classes</b>: </br>
generate_input_data.py: Generates input data for training EmberGAN </br></br>
test_ember_functions.py: Contains methods to predict, score and retrain EmberNet </br></br>
api_module_mapping.py: Contains method to determine which import module an api should be appended to when generating the adversarial samples. However, does not give a significant increase in EmberGAN's performance as compared to adding the suggested api into the first import module. Currently not in use by MalGAN_ember.py but code to use it has been left in comments.</br></br>

