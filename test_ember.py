import ember
from keras.models import load_model
import pandas as pd
import numpy as np
from features import PEFeatureExtractor
 

modelpath = "../../ember_dataset/model.h5"
#file_data = "adversarial_ember_samples.jsonl"
raw_feature_paths = ["adversarial_ember_samples_3.jsonl"]
X_path = "X_adversarial_test.dat"
y_path = "y_adversarial_test.dat"

#ember.vectorize_subset(X_path, y_path, raw_feature_paths, 369)

#load X and y from .dat files
ndim = 128
X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(369, ndim))
y = np.memmap(y_path, dtype=np.float32, mode="r", shape=369)

#separate X into 8 feature arrays
X = embernet.separate_by_feature(X)

model = load_model(modelpath)

y_pred = model.predict(X).reshape(len(y),)
print("y_pred:",y_pred)
acc = model.evaluate(X, y)[1]
print("acc:",acc)

def separate_by_feature(self, data):
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

