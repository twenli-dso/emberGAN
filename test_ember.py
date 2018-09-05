import ember
from keras.models import load_model
import pandas as pd
import numpy as np
from features import PEFeatureExtractor
from sklearn import preprocessing
import multiprocessing
import os
import pickle
 
#scale and separate dataset by feature group
def scale_features(data_dir, X, y):
    # Read vectorized features
    emberdf = ember.read_metadata(data_dir)

    # Combine
    #X = np.concatenate((X_train, X_test))
    #y = np.concatenate((y_train, y_test))
    df = pd.concat([emberdf, pd.DataFrame(X)], axis=1)
    print(df.shape)

    # Save combined dataframe to pickle
    #df.to_pickle(os.path.join(data_dir, "df_test.pkl"))

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

    '''
    # Save to pickle
    print("Saving scalers to scalers.pickle...")
    pickle_out = open(os.path.join(data_dir, 'scalers.pickle'), 'wb')
    pickle.dump(scaler_dict, pickle_out)
    print("Saving scaled data to df_scaled.pkl")
    df_scaled.to_pickle(os.path.join(data_dir, 'df_scaled.pkl'))
    '''

    return df_scaled.iloc[:, 4:].values, df.label

def separate_by_feature(data):
        """
        Separate data by feature group to feed multiple input model
        """
        extractor = PEFeatureExtractor()
        dim = 256
        f_dict = {}
        end = 0

        for feature in extractor.features:
            start = end
            end += feature.dim
            f_dict[feature.name] = data[..., start:end]

        if data.ndim==1:
            return [f_dict['histogram'].reshape(1, dim, 1), f_dict['byteentropy'].reshape(1, dim, 1), np.expand_dims(f_dict['strings'], axis=0), np.expand_dims(f_dict['general'], axis=0), np.expand_dims(f_dict['header'], axis=0), np.expand_dims(f_dict['section'], axis=0), np.expand_dims(f_dict['imports'], axis=0), np.expand_dims(f_dict['exports'], axis=0)]

        data_len = len(data)
        return [f_dict['histogram'].reshape(data_len, dim, 1), f_dict['byteentropy'].reshape(data_len, dim, 1), f_dict['strings'], f_dict['general'], f_dict['header'], f_dict['section'], f_dict['imports'], f_dict['exports']]

def create_metadata():
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    feature_paths = ["adversarial_ember_samples_3.jsonl"]
    records = list(pool.imap(ember.read_metadata_record, ember.raw_feature_iterator(feature_paths)))
    records = [dict(record, **{"subset": "test"}) for record in records]

    metadf = pd.DataFrame(records)[["sha256", "appeared", "subset", "label"]]
    metadf.to_csv("metadata_test.csv")
    return metadf

#create_metadata()

modelpath = "../../ember_dataset/model.h5"
#file_data = "adversarial_ember_samples.jsonl"
raw_feature_paths = ["adversarial_ember_samples_3.jsonl"]
X_path = "X_adversarial_test.dat"
y_path = "y_adversarial_test.dat"
data_dir = "../../ember_dataset"

#ember.vectorize_subset(X_path, y_path, raw_feature_paths, 369)

#load X and y from .dat files
ndim = 256
X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(369, ndim))
y = np.memmap(y_path, dtype=np.float32, mode="r", shape=369)

#scale 
#X, y = scale_features("../../ember_dataset", X, y)

# Retrieve scalers used on train set
pickle_in = open(os.path.join(data_dir, 'scalers.pickle'), 'rb')
scaler_dict = pickle.load(pickle_in)

#test scaling on one sample
first_feature_vector = X[0]

print("Before scaling")
scaled_feature_vector = np.copy(first_feature_vector)
print(scaled_feature_vector)
print(scaled_feature_vector.shape)

# Scale each feature group using scalers fitted to train set
extractor = PEFeatureExtractor()
end = 0
for feature in extractor.features:
    scaler = scaler_dict[feature.name]
    start = end
    end += feature.dim
    scaled_feature_vector[..., start:end] = scaler.transform(first_feature_vector[..., start:end].reshape(1, -1))
print("After scaling", scaled_feature_vector)
print(scaled_feature_vector.shape)

sample = separate_by_feature(scaled_feature_vector)

model = load_model(modelpath)

print("model.predict(sample):",model.predict(sample))

'''
#separate X into 8 feature arrays
X = separate_by_feature(X)

model = load_model(modelpath)

y_pred = model.predict(X).reshape(len(y),)
print("y_pred:",y_pred)
acc = model.evaluate(X, y)[1]
print("acc:",acc)
'''