import ember
from keras.models import load_model
import pandas as pd
import numpy as np
import json
from features import PEFeatureExtractor
from sklearn import preprocessing
import multiprocessing
import os
import pickle

modelpath = "../../ember_dataset/model.h5"
data_dir = "../../ember_dataset"
raw_feature_paths = ["original_malware_samples_8192.jsonl"]
#X_path = "X_adversarial_test.dat"
#y_path = "y_adversarial_test.dat"
X_path = "X_orig_malware_test.dat"
y_path = "y_orig_malware_test.dat"
num_samples = 1203

ember.vectorize_subset(X_path, y_path, raw_feature_paths, num_samples)

#load X and y from .dat files
ndim = PEFeatureExtractor.dim
print(X_path)
X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)

#print(X[0].shape)
scores = []
for i in range(num_samples):
    score = ember.predict_samplevector(modelpath, X[i])
    scores.add(score)
    if i % 10 == 0:
        print("calculating score for sample:",i)

scores_dict = dict(Counter(scores))
print("scores_dict:",scores_dict)
with open("scores_dict_orig_malware.json", "w") as outfile:
    json.dump(scores_dict, outfile)