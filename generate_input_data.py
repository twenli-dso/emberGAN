import json, glob, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Extract samples from original jsonl feature file from ember into samples_n directory, split into malware and benign jsonl files.
def extract_n_samples(n, emberfp):
    #create samples_n directory if doesn't exist
    samples_dir = "./samples_%s" % (n)
    os.makedirs(samples_dir, exist_ok=True)
    
    num_benign = int(n * 0.2)
    num_malware = int(n * 0.8)
    
    counter = 0
    jsonArray = []
    with open(emberfp, 'r') as emberfile:
        for line_num, line in enumerate(emberfile):
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

def get_apistats(jsonl_dir):
    filepaths = glob.glob(os.path.join(jsonl_dir, '*.jsonl'))
    apistats_list = []
    
    for filepath in filepaths:
        #print(filepath)
        with open (filepath,"r") as infile:
            for line_num, line in enumerate(infile):

                apistats_dict = {}

                jsonline = json.loads(line)

                sha256 = jsonline["sha256"]
                label = jsonline["label"]
                if label == 0:
                    label = "benign"
                else:
                    label = "malware"

                apistats_dict["name"] = sha256
                apistats_dict["class"] = label

                #retrieve api features and count occurrence of each api
                imports = jsonline["imports"]
                imports = list(imports.values())
                imports_flattened = [item for sublist in imports for item in sublist]
                imports_dict = dict(Counter(imports_flattened))

                apistats_dict["apistats"] = imports_dict

                apistats_list.append(apistats_dict)
                
    return apistats_list

#TODO: make test_features.jsonl a global variable
def generate_input_data(jsonl_dir, n, output_filepath):
    #extract samples if samples dir doesn't exist
    if not os.path.exists(jsonl_dir):
        extract_n_samples(n, "../../ember_dataset/test_features.jsonl")

    apistats_list = get_apistats(jsonl_dir)
    select_number = 128

    apis = []
    for apistats_dict in apistats_list:
        capis = apistats_dict['apistats']
        for api in capis.keys():
            if api not in apis:
                apis.append(api)

    n_samples = len(apistats_list)
    n_features = len(apis)
    loc = {}
    for i in range(n_features):
        loc[apis[i]] = i

    print("n_features: ", n_features)
    x = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, ))
    sha256_names = []
    for i in range(n_samples):
        apistats_dict = apistats_list[i]
        capis = apistats_dict['apistats']
        cls = apistats_dict['class']
        if cls == 'malware':
            y[i] = 1
        for api in capis.keys():
            x[i, loc[api]] = 1

        sha256_names.append(apistats_dict['name'])
    sha256_names = np.array(sha256_names)

    feat_labels = apis   #特征列名
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

    return (xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels)

#jsonl_dir = "./samples"
#(xmal, ymal), (xben, yben), (mal_names, ben_names), (feat_labels) = generate_input_data(jsonl_dir)
#print("xben:",xben)