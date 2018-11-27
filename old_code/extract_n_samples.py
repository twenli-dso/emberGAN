import json
import os

#emberfp represents jsonl feature file
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
            
extract_n_samples(8192, "C:/Users/TWenLi/Downloads/test_features.jsonl")