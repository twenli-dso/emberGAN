
# generator : 输入层维数：128（特征维数）+20（噪声维数）   隐层数：256  输出层：128
# subsititude detector: 128 - 256 - 1
import os
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm, tree
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
import json
#from VOTEClassifier import VOTEClassifier

import test_ember_function
import generate_input_data

original_feat_filepath = ""
original_ben_feat_filepath = ""
added_feat_filepath = ""

class MalGAN():
    def __init__(self, blackbox='RF', same_train_data=1, filename='data_test_names.npz'):
        self.apifeature_dims = 128
        self.z_dims = 20
        self.hide_layers = 256
        self.generator_layers = [self.apifeature_dims+self.z_dims, self.hide_layers, self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]
        self.blackbox = blackbox       # RF LR DT SVM MLP VOTE
        self.same_train_data = same_train_data   # MalGAN and the black-boxdetector are trained on same or different training sets
        optimizer = Adam(lr=0.001)
        self.filename = filename

        # Directories and filepaths for blackbox data
        self.jsonl_dir = "./samples/"
        self.blackbox_modelpath = "../../ember_dataset/model.h5"
        self.bl_xtrain_mal_filepath = "./blackbox_data/bl_xtrain_mal.jsonl"
        self.bl_xtest_mal_filepath = "./blackbox_data/bl_xtest_mal.jsonl"
        self.bl_xtrain_ben_filepath = "./blackbox_data/bl_xtrain_ben.jsonl"
        self.bl_xtest_ben_filepath = "./blackbox_data/bl_xtest_ben.jsonl"
        self.bl_adver_mal_filepath = "./blackbox_data/adver_mal.jsonl"

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

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

    def build_blackbox_detector(self):

        if self.blackbox is 'RF':
            blackbox_detector = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
        elif self.blackbox is 'SVM':
            blackbox_detector = svm.SVC()
        elif self.blackbox is 'LR':
            blackbox_detector = linear_model.LogisticRegression()
        elif self.blackbox is 'DT':
            blackbox_detector = tree.DecisionTreeRegressor()
        elif self.blackbox is 'MLP':
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                              solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                              learning_rate_init=.1)
        elif self.blackbox is 'VOTE':
            blackbox_detector = VOTEClassifier()

        return blackbox_detector

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
        
        return generate_input_data.generate_input_data(self.jsonl_dir)
        '''
        data = np.load(self.filename)
        xmal, ymal, xben, yben, mal_names, ben_names, feat_labels = data['xmal'], data['ymal'], data['xben'], data['yben'], data['mal_names'], data['ben_names'], data['selected_feat_labels']
        return (xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels)
        '''

    def generate_blackbox_data(self, train_mal_indices, test_mal_indices, train_ben_indices, test_ben_indices):
        #save bl_xtrain_mal etc into jsonl files
        #same_train_data, is_first?
        with open(self.jsonl_dir + "malware_samples_48.jsonl", 'r') as malfile:
            bl_xtrain_mal = []
            bl_xtest_mal = []
            for line_num, line in enumerate(malfile):
                jsonline = json.loads(line)
                if line_num in train_mal_indices:
                    bl_xtrain_mal.append(jsonline)
                elif line_num in test_mal_indices:
                    bl_xtest_mal.append(jsonline)

        with open(self.jsonl_dir + "benign_samples_16.jsonl", 'r') as benfile:
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

        # print("bl_xtrain_mal:",bl_xtrain_mal)
        # print("bl_xtest_mal:",bl_xtest_mal)
        # print("bl_xtrain_mal size:",len(bl_xtrain_mal))
        # print("bl_xtest_mal size:",len(bl_xtest_mal))

    def generate_adversarial_blackbox_data(self, gen_examples, orig_mal, mal_names, feat_labels):
        # TODO: Append added features to blackbox data
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

        #find xmal_batch in blackbox data
        #find by name or idx? 
        with open(self.jsonl_dir + "malware_samples_48.jsonl", 'r') as malfile:
            jsonAdverArray = []
            for line_num, line in enumerate(malfile):
                jsonline = json.loads(line)
                name = jsonline['sha256']
                if name in added_features_dict:
                    added_features = added_features_dict[name]
                    imports = jsonline["imports"]
                    if len(imports) > 0:
                        #add new features to first import module
                        first_module_imports = list(imports.values())[0]  #imports[list(imports.keys())[0]]
                        first_module_imports.extend(added_features)
                        imports[list(imports.keys())[0]] = first_module_imports
                        jsonline["imports"] = imports
                        jsonAdverArray.append(jsonline)

        #print("jsonAdverArray:",jsonAdverArray)

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

        #GENERATE BLACKBOX DATA (and save to jsonl file)
        self.generate_blackbox_data(train_mal_indices, test_mal_indices, train_ben_indices, test_ben_indices)
        bl_ytrain_mal, bl_ytrain_ben, bl_ytest_mal, bl_ytest_ben = ytrain_mal, ytrain_ben, ytest_mal, ytest_ben

        '''
        if self.same_train_data:
            bl_xtrain_mal, bl_ytrain_mal, bl_xtrain_ben, bl_ytrain_ben = xtrain_mal, ytrain_mal, xtrain_ben, ytrain_ben
        else:
            xtrain_mal, bl_xtrain_mal, ytrain_mal, bl_ytrain_mal = train_test_split(xtrain_mal, ytrain_mal, test_size=0.50)
            xtrain_ben, bl_xtrain_ben, ytrain_ben, bl_ytrain_ben = train_test_split(xtrain_ben, ytrain_ben, test_size=0.50)

        # if is_first is Ture, Train the blackbox_detctor
        if is_first:
            self.blackbox_detector.fit(np.concatenate([xmal, xben]),
                                       np.concatenate([ymal, yben]))
        '''

        ytrain_ben_blackbox = test_ember_function.predict(self.blackbox_modelpath, self.bl_xtrain_ben_filepath, len(xtrain_ben))
        Original_Train_TPR = test_ember_function.score(self.blackbox_modelpath, self.bl_xtrain_mal_filepath, bl_ytrain_mal)
        Original_Test_TPR = test_ember_function.score(self.blackbox_modelpath, self.bl_xtest_mal_filepath, bl_ytest_mal)
        print("ytrain_ben_blackbox:", ytrain_ben_blackbox)
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

                ymal_batch = test_ember_function.predict(self.blackbox_modelpath, self.bl_adver_mal_filepath, len(xmal_batch))
                print("ymal_batch:",ymal_batch)

                # Train the substitute_detector
                d_loss_real = self.substitute_detector.train_on_batch(gen_examples, ymal_batch) 
                d_loss_fake = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # print("d_loss_real:",d_loss_real)
                # print("d_loss_fake:",d_loss_fake)
                # print("d_loss:", d_loss)
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
            TPR = test_ember_function.score(self.blackbox_modelpath, self.bl_adver_mal_filepath, bl_ytrain_mal)
            #TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytrain_mal)
            Train_TPR.append(TPR)

            # Compute Test TPR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            self.generate_adversarial_blackbox_data(gen_examples, xtest_mal, test_mal_names, feat_labels)
            TPR = test_ember_function.score(self.blackbox_modelpath, self.bl_adver_mal_filepath, bl_ytest_mal)
            #TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytest_mal)
            Test_TPR.append(TPR)

            # Save best model
            if TPR < best_TPR:
                self.combined.save_weights('saves/malgan.h5')
                best_TPR = TPR

            # Plot the progress
            if is_first:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # ---------------------
        #  Save added features and original features 
        # ---------------------
        
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
        # --------------------------

        flag = ['DiffTrainData', 'SameTrainData']
        print('\n\n---{0} {1}'.format(self.blackbox, flag[self.same_train_data]))
        print('\nOriginal_Train_TPR: {0}, Adver_Train_TPR: {1}'.format(Original_Train_TPR, Train_TPR[-1]))
        print('\nOriginal_Test_TPR: {0}, Adver_Test_TPR: {1}'.format(Original_Test_TPR, Test_TPR[-1]))
        print('\nFirst 10 Test_TPR:',Test_TPR[1:10])
        print('\nLast 10 Test_TPR:',Test_TPR[-10:-1])
        #print('np.ones(gen_examples.shape) * (gen_examples > 0.5):',np.ones(gen_examples.shape) * (gen_examples > 0.5))
        # Plot TPR
        '''
        plt.figure()
        plt.plot(range(len(Train_TPR)), Train_TPR, c='r', label='Training Set', linewidth=2)
        plt.plot(range(len(Test_TPR)), Test_TPR, c='g', linestyle='--', label='Validation Set', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig('saves/Epoch_TPR({0}, {1}).png'.format(self.blackbox, flag[self.same_train_data]))
        plt.show()
        '''

    def retrain_blackbox_detector(self):
        (xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels) = self.load_data()
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.20)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.20)
        # Generate Train Adversarial Examples
        noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtrain_mal, noise])
        gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.5)
        self.blackbox_detector.fit(np.concatenate([xtrain_mal, xtrain_ben, gen_examples]),
                                   np.concatenate([ytrain_mal, ytrain_ben, ytrain_mal]))

        # Compute Train TPR
        train_TPR = self.blackbox_detector.score(gen_examples, ytrain_mal)

        # Compute Test TPR
        noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtest_mal, noise])
        gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.5)
        test_TPR = self.blackbox_detector.score(gen_examples, ytest_mal)
        print('\n---TPR after the black-box detector is retrained(Before Retraining MalGAN).')
        print('\nTrain_TPR: {0}, Test_TPR: {1}'.format(train_TPR, test_TPR))

if __name__ == '__main__':

    original_feat_filepath = "./feature_dicts/original_features_dict_%s.json" % (blackbox)
    original_ben_feat_filepath = "./feature_dicts/original_ben_features_dict_%s.json" % (blackbox)
    added_feat_filepath = "./feature_dicts/added_features_dict_%s.json" % (blackbox)

    malgan = MalGAN()
    malgan.train(epochs=100, batch_size=8)
    malgan.retrain_blackbox_detector()
    malgan.train(epochs=20, batch_size=8, is_first=False)
    '''
    for i in range(10):
        malgan.retrain_blackbox_detector()
        malgan.train(epochs=100, batch_size=64, is_first=False)
    '''
