import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"

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

def predict(model, modelpath, scaler, raw_feature_path, num_samples):
    X_path = "X_test_2.dat"
    y_path = "y_test_2.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    #print(X_path)
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    '''
    scores = []
    for i in range(num_samples):
        score = ember.predict_samplevector(modelpath, X[i])
        scores.append(score[0][0])
        print(score[0][0])
    '''
    
    scores = ember.predict_samplevector(model, modelpath, scaler, X)
    scores = scores.flatten()
    return np.around(scores)


def score(model, modelpath, scaler, raw_feature_path, actual_labels):
    num_samples = len(actual_labels)
    predicted_labels = predict(model, modelpath, scaler, raw_feature_path, len(actual_labels))
    diff = np.subtract(predicted_labels, actual_labels)
    num_wrong = np.count_nonzero(diff)
    TPR = (num_samples - num_wrong) / num_samples

    return TPR

def retrain(model, scaler, raw_feature_path, num_samples):
    X_path = "X_test_retrain.dat"
    y_path = "y_test_retrain.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    #print(X_path)
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    epochs = 50
    batch_size = 64
    retrained_model = ember.retrain_model(model, scaler, X, y, epochs, batch_size)
    retrained_model.save("./blackbox_data/adver/retrained_model.h5")

    return retrained_model
'''
model = load_model("./blackbox_data/adver/model.h5")
pickle_in = open("../../ember_dataset/scalers.pickle", "rb")
scaler = pickle.load(pickle_in)
raw_feature_path = "./blackbox_data/adver/adver_mal.jsonl"
retrain(model, scaler, raw_feature_path, int(0.2*8192))
'''
