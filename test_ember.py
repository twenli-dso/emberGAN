import ember
from keras.models import load_model
import pandas as pd
import numpy as np
from embernet import separate_by_feature

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
X = separate_by_feature(X)

model = load_model(modelpath)

y_pred = model.predict(X).reshape(len(y),)
print("y_pred:",y_pred)
acc = model.evaluate(X, y)[1]
print("acc:",acc)

