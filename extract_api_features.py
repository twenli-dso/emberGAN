import os
import ember
import numpy as np
import pandas as pd


data_dir = "../../ember_dataset/"

#emberdf = ember.create_metadata(data_dir)
emberdf = ember.read_metadata(data_dir)
#X_train, y_train, X_test, y_test = ember.create_vectorized_features(data_dir)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
df = pd.concat([emberdf, pd.DataFrame(X)], axis=1)

print(df.head)

#mal_sha256 = df[df.label == 1].sha256


