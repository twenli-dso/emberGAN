import ember

modelpath = "../../ember_dataset/model.h5"
filedata = "adversarial_ember_samples.json"
raw_feature_paths = ["adversarial_ember_samples.json"]
X_path = "X_adversarial_test.dat"
y_path = "y_adversarial_test.dat"

ember.vectorize_subset(X_path, y_path, raw_feature_paths, 4096)

ember.predict_sample(modelpath, file_data)