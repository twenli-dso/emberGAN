import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def predict(modelpath, raw_feature_paths):
    X_path = "X_test_2.dat"
    y_path = "y_test_2.dat"
    #num_samples = 1203
    num_samples = 369

    ember.vectorize_subset(X_path, y_path, raw_feature_paths, num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    #print(X_path)
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)

    #print(X[0].shape)
    scores = []
    for i in range(10):
        score = ember.predict_samplevector(modelpath, X[i])
        scores.append(score)
        print(score)

    #TO DO: convert scores to binary

    return scores

modelpath = "../../ember_dataset/model.h5"
raw_feature_paths = ["original_malware_samples.jsonl"]

labels = predict(modelpath, raw_feature_paths)
print("labels: ", labels)
