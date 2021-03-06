
# generator : 输入层维数：128（特征维数）+20（噪声维数）   隐层数：256  输出层：128
# subsititude detector: 128 - 256 - 1
import os
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

from sklearn.model_selection import train_test_split
import numpy as np
import json
import pickle
import csv

import test_ember_functions
import generate_input_data
import api_module_mapping

original_feat_filepath = ""
original_ben_feat_filepath = ""
added_feat_filepath = ""
iter_num = 0
TPR_list = []

class MalGAN():
    def __init__(self, num_samples=8192):
        self.apifeature_dims = 512
        self.z_dims = 20
        self.hide_layers = 256
        self.generator_layers = [self.apifeature_dims+self.z_dims, self.hide_layers, self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]
        optimizer = Adam(lr=0.001)

        # Directories and filepaths for blackbox data
        self.blackbox_num_samples = num_samples
        self.data_filepath = 'data_ember_%s.npz' % (num_samples)
        self.jsonl_dir = "./samples_%s/" % (self.blackbox_num_samples)
        self.mal_samples_filepath = "%smalware_samples_%s.jsonl" % (self.jsonl_dir, int(self.blackbox_num_samples * 0.8))
        self.ben_samples_filepath = "%sbenign_samples_%s.jsonl" % (self.jsonl_dir, int(self.blackbox_num_samples * 0.2))
        self.blackbox_modelpath = "../../ember_dataset/model.h5"
        self.blackbox_model = load_model(self.blackbox_modelpath)
        self.ember_filepath = "../../ember_dataset/test_features.jsonl"
        self.bl_xtrain_mal_filepath = "./blackbox_data/bl_xtrain_mal.jsonl"
        self.bl_xtest_mal_filepath = "./blackbox_data/bl_xtest_mal.jsonl"
        self.bl_xtrain_ben_filepath = "./blackbox_data/bl_xtrain_ben.jsonl"
        self.bl_xtest_ben_filepath = "./blackbox_data/bl_xtest_ben.jsonl"
        self.bl_adver_mal_filepath = "./blackbox_data/adver_mal.jsonl"
        
        # Load scaler used for training ember
        data_dir = os.path.dirname(self.blackbox_modelpath)
        pickle_in = open(os.path.join(data_dir, 'scalers.pickle'), 'rb')
        self.scaler = pickle.load(pickle_in)

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation='sigmoid')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation='sigmoid')(x)
        substitute_detector = Model(input, x, name='substitute_detector')
        substitute_detector.summary()
        return substitute_detector

    def load_data(self):
        """ Generates or reads data used for training and testing of GAN"""
        if not os.path.exists(self.data_filepath):
            print("Generating input data...")
            generate_input_data.generate_input_data(self.jsonl_dir, self.blackbox_num_samples, iter_num, self.data_filepath, self.ember_filepath)
        data = np.load(self.data_filepath)
        xmal, ymal, xben, yben, mal_names, ben_names, selected_feat_labels = data['xmal'], data['ymal'], data['xben'], data['yben'], data['mal_names'], data['ben_names'], data['selected_feat_labels']
        return (xmal, ymal), (xben, yben), (mal_names, ben_names), (selected_feat_labels)
        #return generate_input_data.generate_input_data(self.jsonl_dir, self.blackbox_num_samples, iter_num, 'data_ember_%s.npz' % (self.blackbox_num_samples), self.ember_filepath)

    def generate_blackbox_data(self, train_mal_indices, test_mal_indices, train_ben_indices, test_ben_indices):
        # Save bl_xtrain_mal etc into jsonl files

        with open(self.mal_samples_filepath, 'r') as malfile:
            bl_xtrain_mal = []
            bl_xtest_mal = []
            for line_num, line in enumerate(malfile):
                jsonline = json.loads(line)
                if line_num in train_mal_indices:
                    bl_xtrain_mal.append(jsonline)
                elif line_num in test_mal_indices:
                    bl_xtest_mal.append(jsonline)

        with open(self.ben_samples_filepath, 'r') as benfile:
            bl_xtrain_ben = []
            bl_xtest_ben = []
            for line_num, line in enumerate(benfile):
                jsonline = json.loads(line)
                if line_num in train_ben_indices:
                    bl_xtrain_ben.append(jsonline)
                elif line_num in test_ben_indices:
                    bl_xtest_ben.append(jsonline)

        with open(self.bl_xtrain_mal_filepath, 'w') as outfile:
            for jsonline in bl_xtrain_mal:
                json.dump(jsonline, outfile)
                outfile.write('\n')

        with open(self.bl_xtest_mal_filepath, 'w') as outfile:
            for jsonline in bl_xtest_mal:
                json.dump(jsonline, outfile)
                outfile.write('\n')

        with open(self.bl_xtrain_ben_filepath, 'w') as outfile:
            for jsonline in bl_xtrain_ben:
                json.dump(jsonline, outfile)
                outfile.write('\n')

        with open(self.bl_xtest_ben_filepath, 'w') as outfile:
            for jsonline in bl_xtest_ben:
                json.dump(jsonline, outfile)
                outfile.write('\n')

    def generate_adversarial_blackbox_data(self, gen_examples, orig_mal, mal_names, feat_labels):
        #Extract added features
        new_examples = np.ones(gen_examples.shape)*(gen_examples > 0.5)
        added_features = np.subtract(new_examples, orig_mal)
        added_features_labels = []
        for added_feature in added_features:
            added_feature_labels = feat_labels[np.where(added_feature == 1)]
            added_features_labels.append(added_feature_labels)
        
        #print("added_features_labels:",added_features_labels)

        added_features_dict = {}
        for i, mal_name in enumerate(mal_names):
            added_features_dict[mal_name] = added_features_labels[i].tolist()

        #print("added_features_dict:",added_features_dict)

        '''
        #load api to module mapping or generate it if doesn't exist
        try:
            with open("./api_module_mapping/api_module_mapping_%s.json" % (self.blackbox_num_samples), "r") as infile:
                api_module_dict = json.load(infile)
        except FileNotFoundError:
                api_module_dict = api_module_mapping.gen_api_module_mapping(self.blackbox_num_samples)
        '''

        # Append added features into original json file
        with open(self.mal_samples_filepath, 'r') as malfile:
            jsonAdverArray = []
            for line_num, line in enumerate(malfile):
                jsonline = json.loads(line)
                name = jsonline['sha256']
                if name in added_features_dict:
                    added_features = added_features_dict[name]

                    #check if added feature belongs to imports or header characteristics or section properties
                    for added_feature in added_features:
                        category = added_feature.split(":")[0]

                        imports = jsonline["imports"]
                        if category == "imports" and len(imports) > 0:
                            feature = added_feature.split(":")[1]
                            '''
                            #add new features to mapped module if exists, otherwise insert into first module
                            mapped_module = api_module_dict[added_feature]
                            
                            if mapped_module in imports:
                                mapped_module_imports = imports[mapped_module]
                                mapped_module_imports.append(added_feature)
                                imports[mapped_module] = mapped_module_imports
                            else:
                            '''
                            #add to first module if mapped_module does not exist for this malfile
                            first_module_imports = list(imports.values())[0]  #imports[list(imports.keys())[0]]
                            first_module_imports.append(feature)
                            imports[list(imports.keys())[0]] = first_module_imports
                        
                            jsonline["imports"] = imports

                        elif category == "chars":
                            feature = added_feature.split(":")[1]

                            header_chars = jsonline['header']['coff']['characteristics']
                            header_chars.append(feature)
                            jsonline['header']['coff']['characteristics'] = header_chars

                        elif category == "section_props":
                            section_name = added_feature.split(":")[1]
                            feature = added_feature.split(":")[2]

                            sections = jsonline['section']['sections']
                            for section in sections:
                                if section['name'] == section_name:
                                    section['props'].append(feature) 
                            jsonline['section']['sections'] = sections

                    jsonAdverArray.append(jsonline)

        with open(self.bl_adver_mal_filepath, 'w') as outfile:
            for jsonline in jsonAdverArray:
                json.dump(jsonline, outfile)
                outfile.write('\n')

    def train(self, epochs, batch_size=32, is_first=1):

        # Load and Split the dataset
        (xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels) = self.load_data()
        print("xmal shape:",xmal.shape)
        print("xben shape:",xben.shape)

        mal_indices = np.arange(xmal.shape[0])
        ben_indices = np.arange(xben.shape[0])

        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal, train_mal_indices, test_mal_indices = train_test_split(xmal, ymal, mal_indices, test_size=0.20)
        train_mal_names = mal_names[train_mal_indices]
        test_mal_names = mal_names[test_mal_indices]

        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben, train_ben_indices, test_ben_indices = train_test_split(xben, yben, ben_indices, test_size=0.20)
        train_ben_names = ben_names[train_ben_indices]
        test_ben_names = ben_names[test_ben_indices]

        # GENERATE BLACKBOX DATA (and save to jsonl file)
        self.generate_blackbox_data(train_mal_indices, test_mal_indices, train_ben_indices, test_ben_indices)
        bl_ytrain_mal, bl_ytrain_ben, bl_ytest_mal, bl_ytest_ben = ytrain_mal, ytrain_ben, ytest_mal, ytest_ben

        # Calculate original TPR
        ytrain_ben_blackbox = test_ember_functions.predict(self.blackbox_model, self.scaler, self.bl_xtrain_ben_filepath, len(xtrain_ben))
        Original_Train_TPR = test_ember_functions.score(self.blackbox_model, self.scaler, self.bl_xtrain_mal_filepath, bl_ytrain_mal)
        Original_Test_TPR = test_ember_functions.score(self.blackbox_model, self.scaler, self.bl_xtest_mal_filepath, bl_ytest_mal)
        #print("ytrain_ben_blackbox:", ytrain_ben_blackbox)
        print("Original_Train_TPR:",Original_Train_TPR)
        print("Original_Test_TPR:",Original_Test_TPR)

        Train_TPR, Test_TPR = [Original_Train_TPR], [Original_Test_TPR]
        best_TPR = 1.0
        for epoch in range(epochs):
            for step in range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                # Select a random batch of malware examples
                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                xmal_batch_names = train_mal_names[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))
                idx = np.random.randint(0, xmal_batch.shape[0], batch_size)
                xben_batch = xtrain_ben[idx]
                yben_batch = ytrain_ben_blackbox[idx]

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                self.generate_adversarial_blackbox_data(gen_examples, xmal_batch, xmal_batch_names, feat_labels)

                ymal_batch = test_ember_functions.predict(self.blackbox_model, self.scaler, self.bl_adver_mal_filepath, len(xmal_batch))

                # Train the substitute_detector
                d_loss_real = self.substitute_detector.train_on_batch(gen_examples, ymal_batch) 
                d_loss_fake = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                # Train the generator
                g_loss = self.combined.train_on_batch([xmal_batch, noise], np.zeros((batch_size, 1)))
                #print("[xmal_batch, noise], np.zeros((batch_size, 1)):",[xmal_batch, noise], np.zeros((batch_size, 1)))

            # Compute Train TPR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtrain_mal, noise])
            self.generate_adversarial_blackbox_data(gen_examples, xtrain_mal, train_mal_names, feat_labels)
            TPR = test_ember_functions.score(self.blackbox_model, self.scaler, self.bl_adver_mal_filepath, bl_ytrain_mal)
            print("Train_TPR:",TPR)
            Train_TPR.append(TPR)

            # Compute Test TPR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            self.generate_adversarial_blackbox_data(gen_examples, xtest_mal, test_mal_names, feat_labels)
            TPR = test_ember_functions.score(self.blackbox_model, self.scaler, self.bl_adver_mal_filepath, bl_ytest_mal)
            print("Test_TPR:",TPR)
            Test_TPR.append(TPR)

            # Save best model
            if TPR < best_TPR:
                self.combined.save_weights('saves/malgan.h5')
                best_TPR = TPR

            # Plot the progress
            if is_first:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                with open("./MalGAN_ember_mapped_%s.txt" % (self.blackbox_num_samples) , "a") as outfile:
                    outfile.write("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]\n" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # -------------------------------------------------------------------------
        #  Save added features and original features 
        # -------------------------------------------------------------------------
        
        #Extract added features
        new_examples = np.ones(gen_examples.shape)*(gen_examples > 0.5)
        added_features = np.subtract(new_examples, xtest_mal)
        added_features_labels = []
        for added_feature in added_features:
            added_feature_labels = feat_labels[np.where(added_feature == 1)]
            added_features_labels.append(added_feature_labels)
        #print('added_features_labels[0]:',added_features_labels[0])

        added_features_dict = {}
        for i, mal_name in enumerate(test_mal_names):
            added_features_dict[mal_name] = added_features_labels[i].tolist()
        #print('added_features_dict:',added_features_dict)

        with open(added_feat_filepath, 'w') as outfile:
            json.dump(added_features_dict, outfile)

        #Save original features in dict
        original_features_labels = []
        for original_feature in xtest_mal:
            original_feature_labels = feat_labels[np.where(original_feature == 1)]
            original_features_labels.append(original_feature_labels)

        original_features_dict = {}
        for i, mal_name in enumerate(test_mal_names):
            original_features_dict[mal_name] = original_features_labels[i].tolist()

        with open(original_feat_filepath, 'w') as outfile:
            json.dump(original_features_dict, outfile)
        
        #Save ben features to file
        original_benign_features_labels = []
        for original_benign_feature in xtest_ben:
            original_benign_feature_labels = feat_labels[np.where(original_benign_feature == 1)]
            original_benign_features_labels.append(original_benign_feature_labels)

        original_benign_features_dict = {}
        for i, ben_name in enumerate(test_ben_names):
            original_benign_features_dict[ben_name] = original_benign_features_labels[i].tolist()

        with open(original_ben_feat_filepath, 'w') as outfile:
            json.dump(original_benign_features_dict, outfile)
        # ------------------------------------------------------------------

        flag = ['DiffTrainData', 'SameTrainData']
        TPR_list.append(Original_Test_TPR)
        TPR_list.append(Test_TPR[-1])
        print('\nOriginal_Train_TPR: {0}, Adver_Train_TPR: {1}'.format(Original_Train_TPR, Train_TPR[-1]))
        print('\nOriginal_Test_TPR: {0}, Adver_Test_TPR: {1}'.format(Original_Test_TPR, Test_TPR[-1]))
        print('\nFirst 10 Test_TPR:',Test_TPR[1:10])
        print('\nLast 10 Test_TPR:',Test_TPR[-10:-1])
        #print('np.ones(gen_examples.shape) * (gen_examples > 0.5):',np.ones(gen_examples.shape) * (gen_examples > 0.5))
        

    def retrain_blackbox_detector(self, epochs, batch_size):
        # Load and Split the dataset
        (xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels) = self.load_data()
        print("xmal shape:",xmal.shape)
        print("xben shape:",xben.shape)
        mal_indices = np.arange(xmal.shape[0])
        ben_indices = np.arange(xben.shape[0])

        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal, train_mal_indices, test_mal_indices = train_test_split(xmal, ymal, mal_indices, test_size=0.20)
        train_mal_names = mal_names[train_mal_indices]
        test_mal_names = mal_names[test_mal_indices]

        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben, train_ben_indices, test_ben_indices = train_test_split(xben, yben, ben_indices, test_size=0.20)
        train_ben_names = ben_names[train_ben_indices]
        test_ben_names = ben_names[test_ben_indices]

        # GENERATE BLACKBOX DATA (and save to jsonl file)
        self.generate_blackbox_data(train_mal_indices, test_mal_indices, train_ben_indices, test_ben_indices)
        bl_ytrain_mal, bl_ytrain_ben, bl_ytest_mal, bl_ytest_ben = ytrain_mal, ytrain_ben, ytest_mal, ytest_ben

        # Generate Train Adversarial Examples
        noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtrain_mal, noise])
        self.generate_adversarial_blackbox_data(gen_examples, xtrain_mal, train_mal_names, feat_labels)

        # Retrain ember with adversarial examples
        retrained_ember = test_ember_functions.retrain(self.blackbox_model, self.scaler, self.bl_adver_mal_filepath, len(xtrain_mal), epochs, batch_size)

        # Compute Train TPR
        train_TPR = test_ember_functions.score(retrained_ember, self.scaler, self.bl_adver_mal_filepath, bl_ytrain_mal)

        # Compute Test TPR
        noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtest_mal, noise])
        self.generate_adversarial_blackbox_data(gen_examples, xtest_mal, test_mal_names, feat_labels)
        test_TPR = test_ember_functions.score(retrained_ember, self.scaler, self.bl_adver_mal_filepath, bl_ytest_mal)
        print('\n---TPR after the black-box detector is retrained(Before Retraining MalGAN).')
        print('\nTrain_TPR: {0}, Test_TPR: {1}'.format(train_TPR, test_TPR))
        TPR_list.append(test_TPR)

if __name__ == '__main__':
    blackbox = 'ember'

    # Filepaths for saving original and generated features
    original_feat_filepath = "./feature_dicts/original_features_dict_%s.json" % (blackbox)
    original_ben_feat_filepath = "./feature_dicts/original_ben_features_dict_%s.json" % (blackbox)
    added_feat_filepath = "./feature_dicts/added_features_dict_%s.json" % (blackbox)

    # Save results of ember into csv file
    with open("compiled_results.csv","w") as csvfile:
        headers = ["Original TPR",	"Adver TPR", "Adver TPR After Retraining EmberNet", "Adver TPR After Retraining EmberNet 2", "Adver TPR After Retraining EmberGAN"]
        csv_writer = csv.writer(csvfile, delimiter = ',')
        csv_writer.writerow(headers)

    for iter_num in range(10):
        with open("compiled_results.csv","a") as csvfile:
            print("----------------RUNNING ITERATION #%s-----------------\n" %(iter_num))
            malgan = MalGAN()
            malgan.train(epochs=10, batch_size=128)
            malgan.retrain_blackbox_detector(epochs=10, batch_size=128)
            malgan.train(epochs=5, batch_size=128)

            csv_writer = csv.writer(csvfile, delimiter = ',')
            csv_writer.writerow(TPR_list)
            TPR_list = []