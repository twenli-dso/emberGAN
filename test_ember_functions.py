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
from ember.features import PEFeatureExtractor
from sklearn import preprocessing
import multiprocessing
import os
import pickle

def predict(model, scaler, raw_feature_path, num_samples):
    """ Get prediction of whether samples are malicious or benign

    Parameters:
        model: EmberNet model file
        scaler: Pickle file with scaler used for training EmberNet
        raw_feature_path: Filepath of samples
        num_samples: Number of samples in file

    Returns:
        scores: List of predictions for each sample (0 for benign, 1 for malicious)
    """

    X_path = "X_test_2.dat"
    y_path = "y_test_2.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    scores = ember.predict_samplevector(model, scaler, X)
    scores = scores.flatten()
    return np.around(scores)


def score(model, scaler, raw_feature_path, actual_labels):
    """ Calculate TPR of model with provided samples

    Parameters:
        model: EmberNet model file
        scaler: Pickle file with scaler used for training EmberNet
        raw_feature_path: Filepath of samples
        actual_labels: Actual label of each sample (0 for benign, 1 for malicious)

    Returns:
        TPR (float): True Positive Rate of EmberNet based on provided samples
    """

    num_samples = len(actual_labels)
    predicted_labels = predict(model, scaler, raw_feature_path, len(actual_labels))
    
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    mal_pos = np.where(actual_labels == 1)
    mal_labels = actual_labels[mal_pos]
    pred_labels_for_mal = predicted_labels[mal_pos]
    diff = np.subtract(mal_labels, pred_labels_for_mal)
    false_negatives = np.count_nonzero(diff)
    total_positives = len(mal_pos[0])
    
    TPR = (total_positives - false_negatives) / total_positives

    return TPR

def retrain(model, scaler, raw_feature_path, num_samples, epochs, batch_size):
    """ Retrain model with new samples

    Parameters:
        model: EmberNet model file
        scaler: Pickle file with scaler used for training EmberNet
        raw_feature_path: Filepath of samples
        num_samples: Number of samples in file
        epochs: Number of training epochs 
        batch_size: Batch size for training

    Returns:
        retrained_model: Model that has been retrained with new samples
    """

    X_path = "X_test_retrain.dat"
    y_path = "y_test_retrain.dat"

    ember.vectorize_subset(X_path, y_path, [raw_feature_path], num_samples)

    #load X and y from .dat files
    ndim = PEFeatureExtractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(num_samples, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=num_samples)
    
    retrained_model = ember.retrain_model(model, scaler, X, y, epochs, batch_size)
    retrained_model.save("./retrained_model.h5")

    return retrained_model

