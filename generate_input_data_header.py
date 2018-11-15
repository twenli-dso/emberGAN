import json, glob, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Extract samples from original jsonl feature file from ember into samples_n directory, split into malware and benign jsonl files.
def extract_n_samples(n, start_index, emberfp):
    #create samples_n directory if doesn't exist
    samples_dir = "./samples_%s" % (n)
    os.makedirs(samples_dir, exist_ok=True)
    
    num_benign = int(n * 0.2)
    num_malware = int(n * 0.8)
    
    counter = 0
    jsonArray = []
    with open(emberfp, 'r') as emberfile:
        for line_num, line in enumerate(emberfile):
            if line_num >= start_index:
                if counter < num_benign:
                    jsonline = json.loads(line)
                    label = jsonline["label"]

                    #add to array if benign
                    if label == 0:
                        jsonArray.append(jsonline)
                        counter += 1
                else:
                    break

    with open('%s/benign_samples_%s.jsonl' % (samples_dir, num_benign), 'w') as outfile:
        for jsonline in jsonArray:
            json.dump(jsonline, outfile)
            outfile.write('\n')
            
    counter = 0
    jsonArray = []
    with open(emberfp, 'r') as emberfile:
        for line_num, line in enumerate(emberfile):
            if line_num >= start_index:
                if counter < num_malware:
                    jsonline = json.loads(line)
                    label = jsonline["label"]

                    #add to array if malware
                    if label == 1:
                        jsonArray.append(jsonline)
                        counter += 1
                else:
                    break

    with open('%s/malware_samples_%s.jsonl' % (samples_dir, num_malware), 'w') as outfile:
        for jsonline in jsonArray:
            json.dump(jsonline, outfile)
            outfile.write('\n')

def get_target_features(jsonl_dir):
    filepaths = glob.glob(os.path.join(jsonl_dir, '*.jsonl'))

    target_features_list = []
    
    for filepath in filepaths:
        #print(filepath)
        with open (filepath,"r") as infile:
            for line_num, line in enumerate(infile):

                if line_num % 10000 == 0:
                    print("processing line_num #", line_num)

                target_features_dict = {}
                target_features = []

                jsonline = json.loads(line)

                sha256 = jsonline["sha256"]
                label = jsonline["label"]
                if label == 0:
                    label = "benign"
                else:
                    label = "malware"

                target_features_dict["name"] = sha256
                target_features_dict["class"] = label

                #retrieve api features
                imports = jsonline["imports"]
                imports = list(imports.values())
                imports_flattened = ["imports:" + item for sublist in imports for item in sublist]
                target_features.extend(imports_flattened)

                '''
                #retrieve header characteristics
                header_chars = jsonline['header']['coff']['characteristics']
                header_chars = ["chars:" + header_char for header_char in header_chars]
                target_features.extend(header_chars)

                #retrieve section props
                sections = jsonline['section']['sections']
                for section in sections:
                    section_name = section['name']
                    props = section['props']
                    section_props = ["section_props:" + section_name + ":" + prop for prop in props]
                    target_features.extend(section_props)
                '''

                target_features_dict['target_features'] = target_features

                target_features_list.append(target_features_dict)

    # print("target_features_list: ", target_features_list[0])
    return target_features_list

#TODO: make test_features.jsonl a global variable
def generate_input_data(jsonl_dir, n, iter_num, output_filepath):
    #extract samples if samples dir doesn't exist
    #if not os.path.exists(jsonl_dir):
    start_index = iter_num * 8192
    extract_n_samples(n, start_index, "../../ember_dataset/test_features.jsonl")

    target_features_list = get_target_features(jsonl_dir)
    select_number = 512

    all_target_features = []
    for target_features_dict in target_features_list:
        target_features = target_features_dict['target_features']
        for target_feature in target_features:
            if target_feature not in all_target_features:
                all_target_features.append(target_feature)
                #if len(all_target_features) >= 2000:
                    #break

    #count all target features and select top 2000 frequent features
    selected_target_features_counts = dict(Counter(all_target_features).most_common(3000))
    selected_target_features = list(selected_target_features_counts.keys())

    n_samples = len(target_features_list)
    n_features = len(selected_target_features)
    #print("n_features: ", n_features)
    #n_features = 2000
    loc = {}
    for i in range(n_features):
        loc[selected_target_features[i]] = i

    x = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, ))
    sha256_names = []
    for i in range(n_samples):
        if i%10000 == 0:
            print("processing sample #", i)
        target_features_dict = target_features_list[i]
        target_features = target_features_dict['target_features']
        cls = target_features_dict['class']
        if cls == 'malware':
            y[i] = 1
        for target_feature in target_features:
            if target_feature in selected_target_features:
                x[i, loc[target_feature]] = 1

        sha256_names.append(target_features_dict['name'])
    sha256_names = np.array(sha256_names)

    feat_labels = selected_target_features   #特征列名
    forest = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1)  #2000棵树,并行工作数是运行服务器决定
    forest.fit(x, y)
    importances = forest.feature_importances_   #feature_importances_特征列重要性占比
    indices = np.argsort(importances)[::-1]     #对参数从小到大排序的索引序号取逆,即最重要特征索引——>最不重要特征索引
    # for f in range(x.shape[1]):
    #     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    #get selected feat_labels from selected indices
    feat_labels = np.array(feat_labels)
    selected_feat_labels = feat_labels[indices[:select_number]]
    #print('selected_feat_labels:',selected_feat_labels)

    x = x[:, indices[:select_number]]
    xmal = x[np.where(y==1)]
    ymal = y[np.where(y==1)]
    mal_names = sha256_names[np.where(y==1)]
    #print("mal_names:",mal_names)

    xben = x[np.where(y==0)]
    yben = y[np.where(y==0)]
    ben_names = sha256_names[np.where(y==0)]
    #print("ben_names:",ben_names)

    np.savez(output_filepath, xmal=xmal, ymal=ymal, xben=xben, yben=yben, mal_names=mal_names, ben_names=ben_names, selected_feat_labels = selected_feat_labels)

    # print("xmal.shape:", xmal.shape)
    # print("ymal.shape:", ymal.shape)
    print("selected_feat_labels:",selected_feat_labels)
    return (xmal, ymal), (xben, yben), (mal_names, ben_names), (selected_feat_labels)

#jsonl_dir = "./samples"
#(xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels) = generate_input_data(jsonl_dir)
#print("xben:",xben)

# get_target_features("C:/Users/TWenLi/Desktop/Ember GAN/samples_8192")
# generate_input_data("C:/Users/TWenLi/Desktop/Ember GAN/samples_8192", 8192, "data_ember_target_features.npz")