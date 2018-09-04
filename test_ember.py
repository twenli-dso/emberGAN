import ember
from keras.models import load_model
import pandas as pd

modelpath = "../../ember_dataset/model.h5"
#file_data = "adversarial_ember_samples.jsonl"
raw_feature_paths = ["adversarial_ember_samples_3.jsonl"]
X_path = "X_adversarial_test.dat"
y_path = "y_adversarial_test.dat"
#df_path = "../../ember_dataset/df_scaled.pkl"

ember.vectorize_subset(X_path, y_path, raw_feature_paths, 369)

model = load_model(modelpath)

print("Loading scaled features...")
df = pd.read_pickle(df_path)
X = df.iloc[:, 4:].values
print("X.shape:",X.shape)
y = df.label
print("y.shape:",y.shape)

y_pred = model.predict(X).reshape(len(y),)
print("y_pred:",y_pred)
acc = model.evaluate(X, y)[1]
print("acc:",acc)

