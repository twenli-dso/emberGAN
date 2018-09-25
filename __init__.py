# -*- coding: utf-8 -*-

import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from .features import PEFeatureExtractor
from .embernet import EmberNet
from sklearn import preprocessing
from keras.models import load_model
import pickle


def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                yield line


def vectorize(irow, raw_features_string, X_path, y_path, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    extractor = PEFeatureExtractor()
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)
    #print("feature_vector.shape:",feature_vector.shape)

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    extractor = PEFeatureExtractor()
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir):
    """
    Create feature vectors from raw features and write them to disk
    """
    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train.dat")
    y_path = os.path.join(data_dir, "y_train.dat")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    vectorize_subset(X_path, y_path, raw_feature_paths, 900000)

    print("Vectorizing test set")
    X_path = os.path.join(data_dir, "X_test.dat")
    y_path = os.path.join(data_dir, "y_test.dat")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    vectorize_subset(X_path, y_path, raw_feature_paths, 200000)


def read_vectorized_features(data_dir, subset=None):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    ndim = PEFeatureExtractor.dim
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(900000, ndim))
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r", shape=900000)
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(200000, ndim))
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=200000)
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test


def read_metadata_record(raw_features_string):
    """
    Decode a raw features stringa and return the metadata fields
    """
    full_metadata = json.loads(raw_features_string)
    return {"sha256": full_metadata["sha256"], "appeared": full_metadata["appeared"], "label": full_metadata["label"]}


def create_metadata(data_dir):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
    train_records = [dict(record, **{"subset": "train"}) for record in train_records]

    test_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
    test_records = [dict(record, **{"subset": "test"}) for record in test_records]

    metadf = pd.DataFrame(train_records + test_records)[["sha256", "appeared", "subset", "label"]]
    metadf.to_csv(os.path.join(data_dir, "metadata.csv"))
    return metadf


def read_metadata(data_dir):
    """
    Read an already created metadata file and return its dataframe
    """
    return pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)


def train_model(data_dir):
    """
    Train the neural network model from the EMBER dataset from the vectorized and scaled features
    """
    embernet = EmberNet()
    modelpath = os.path.join(data_dir, "model.h5")
    df_path = os.path.join(data_dir, "df_scaled.pkl")

    if os.path.isfile(df_path):
        # Load scaled feature vector
        print("Loading scaled features...")
        df = pd.read_pickle(df_path)
        X = df.iloc[:, 4:].values
        print(X)
        y = df.label
        print(y)
    else:
        # Scale vectorized features
        X, y = scale_features(data_dir)

    # Create model if model doesn't exist
    if not os.path.isfile(modelpath):
        print("Creating new model...")
        embernet.create(modelpath)

    # Train
    model = embernet.train(X, y, modelpath, epochs=5)

    # Set threshold for future testing
    print("Finding threshold...")
    embernet.set_threshold(X, y, model, data_dir)

    return model


def predict_sample(modelpath, file_data):
    """
    Predict a PE file with a neural network model
    """
    embernet = EmberNet()
    model = load_model(modelpath)
    data_dir = os.path.dirname(modelpath)

    # Extract features from binary
    extractor = PEFeatureExtractor()
    feature_vector = np.array(extractor.feature_vector(file_data), dtype=np.float32)

    # Retrieve scalers used on train set
    pickle_in = open(os.path.join(data_dir, 'scalers.pickle'), 'rb')
    scaler_dict = pickle.load(pickle_in)

    print("Before scaling")
    scaled_feature_vector = np.copy(feature_vector)
    print(scaled_feature_vector)
    print(scaled_feature_vector.shape)

    # Scale each feature group using scalers fitted to train set
    end = 0
    for feature in extractor.features:
        scaler = scaler_dict[feature.name]
        start = end
        end += feature.dim
        scaled_feature_vector[..., start:end] = scaler.transform(feature_vector[..., start:end].reshape(1, -1))
    print("After scaling", scaled_feature_vector)
    print(scaled_feature_vector.shape)

    sample = embernet.separate_by_feature(scaled_feature_vector)

    return model.predict(sample)

def predict_samplevector(modelpath, vector):
    """
    Predict a PE file with a neural network model
    """
    embernet = EmberNet()
    model = load_model(modelpath)
    data_dir = os.path.dirname(modelpath)

    # Extract features from binary
    extractor = PEFeatureExtractor()
    feature_vector = np.array(vector, dtype=np.float32)

    # Retrieve scalers used on train set
    pickle_in = open(os.path.join(data_dir, 'scalers.pickle'), 'rb')
    scaler_dict = pickle.load(pickle_in)

    #print("Before scaling")
    scaled_feature_vector = np.copy(feature_vector)
    #print(scaled_feature_vector)
    #print(scaled_feature_vector.shape)

    # Scale each feature group using scalers fitted to train set
    end = 0
    for feature in extractor.features:
        scaler = scaler_dict[feature.name]
        start = end
        end += feature.dim
        scaled_feature_vector[..., start:end] = scaler.transform(feature_vector[..., start:end].reshape(1, -1))
    #print("After scaling", scaled_feature_vector)
    #print(scaled_feature_vector.shape)

    sample = embernet.separate_by_feature(scaled_feature_vector)

    return model.predict(sample)

def scale_features(data_dir):
    """
    Scale and separate dataset by feature group
    """
    # Read vectorized features
    emberdf = read_metadata(data_dir)
    X_train, y_train, X_test, y_test = read_vectorized_features(data_dir)

    # Combine
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    df = pd.concat([emberdf, pd.DataFrame(X)], axis=1)
    print(df.shape)

    # Save combined dataframe to pickle
    df.to_pickle(os.path.join(data_dir, "df.pkl"))

    # Remove unlabeled rows
    df = df[df.label != -1]

    # Reset index
    df = df.reset_index(drop=True)

    # Scale by feature group
    print("Scaling each feature group...")
    scaler_dict = dict()
    extractor = PEFeatureExtractor()
    end = 4
    df_scaled = df.iloc[:, :end]
    for feature in extractor.features:
        # scaler = preprocessing.StandardScaler()
        scaler = preprocessing.RobustScaler()
        start = end
        end += feature.dim
        print(feature.name, start, end)

        feature_matrix = df.iloc[:, start:end].as_matrix().astype(float)
        scaler_dict[feature.name] = scaler.fit(feature_matrix)
        scaled_group = scaler.transform(feature_matrix)
        df_scaled = pd.concat([df_scaled, pd.DataFrame(scaled_group)], axis=1)

    df_scaled.columns = df.columns
    print(df_scaled)

    # Save to pickle
    print("Saving scalers to scalers.pickle...")
    pickle_out = open(os.path.join(data_dir, 'scalers.pickle'), 'wb')
    pickle.dump(scaler_dict, pickle_out)
    print("Saving scaled data to df_scaled.pkl")
    df_scaled.to_pickle(os.path.join(data_dir, 'df_scaled.pkl'))

    return df_scaled.iloc[:, 4:].values, df.label


def get_fpr(y_true, y_pred):
    """
    Compute false positive rate
    """
    nbenign = (y_true == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)


def find_threshold(y_true, y_pred, fpr_target):
    """
    Finds threshold for a given false positive rate
    """
    thresh = 0.0
    fpr = get_fpr(y_true, y_pred > thresh)
    while fpr > fpr_target and thresh < 1.0:
        thresh += 0.001
        fpr = get_fpr(y_true, y_pred > thresh)
    return thresh, fpr


def add_data(data_dir, binaries, save_file):
    """
    Add PE files to train set
    """
    extractor = PEFeatureExtractor()

    # Load original dataset
    print("Loading from pickle...")
    df = pd.read_pickle(os.path.join(data_dir, 'df.pkl'))

    for binary_path in binaries:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path))

        file_data = open(binary_path, "rb").read()

        # Extract features from binary
        feature_vector = np.array(extractor.feature_vector(file_data), dtype=np.float32)

        # Append new feature vector
        df = df.append(pd.Series(feature_vector), ignore_index=True)

    print("Updated dataframe:")
    print(df)
    print("Saving to pickle...")
    df.to_pickle(os.path.join(data_dir, save_file))
