import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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

def predict(model, scaler, raw_feature_path, num_samples):
    X_path = "X_test_2.dat"
    y_path = "y_test_2.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    #print(X_path)
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    scores = ember.predict_samplevector(model, scaler, X)
    scores = scores.flatten()
    return np.around(scores)


def score(model, scaler, raw_feature_path, actual_labels):
    num_samples = len(actual_labels)
    predicted_labels = predict(model, scaler, raw_feature_path, len(actual_labels))
    # diff = np.subtract(predicted_labels, actual_labels)
    # num_wrong = np.count_nonzero(diff)
    # TPR = (num_samples - num_wrong) / num_samples
    
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    mal_pos = np.where(actual_labels == 1)
    mal_labels = actual_labels[mal_pos]
    pred_labels_for_mal = predicted_labels[mal_pos]
    diff = np.subtract(mal_labels, pred_labels_for_mal)
    false_negatives = np.count_nonzero(diff)
    total_positives = len(mal_pos[0])
    # print("false_negatives: ", false_negatives)
    # print("total_positives: ", total_positives)
    TPR = (total_positives - false_negatives) / total_positives

    return TPR

def retrain(model, scaler, raw_feature_path, num_samples, epochs, batch_size):
    X_path = "X_test_retrain.dat"
    y_path = "y_test_retrain.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    #print(X_path)
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    retrained_model = ember.retrain_model(model, scaler, X, y, epochs, batch_size)
    retrained_model.save("./blackbox_data/adver/retrained_model.h5")

    return retrained_model

# model = load_model("./blackbox_data/adver/model.h5")
# retrained_model = load_model("./blackbox_data/adver/retrained_model.h5")

# pickle_in = open("../../ember_dataset/scalers.pickle", "rb")
# scaler = pickle.load(pickle_in)
# raw_feature_path = "../../ember_dataset/test_features.jsonl"
# #retrain(model, scaler, raw_feature_path, int(0.2*8192))

# actual_labels = []
# with open(raw_feature_path, "r") as infile:
#     for line_num, line in enumerate(infile):
#         jsonline = json.loads(line)
#         label = jsonline["label"]
#         actual_labels.append(label)

# tpr = score(model, scaler, raw_feature_path, actual_labels)
# print("Original model TPR: ", tpr)

# retrained_tpr = score(retrained_model, scaler, raw_feature_path, actual_labels)
# print("Retrained model TPR: ", retrained_tpr)
