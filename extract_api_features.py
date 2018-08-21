import json
import glob
from collections import Counter

filepaths = glob.glob('../../ember_dataset/*.jsonl')
#filepaths = glob.glob('C:/Users/TWenLi/Downloads/*.jsonl')
for filepath in filepaths:
    print(filepath)
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

            #print(apistats_dict)

            with open('ember_apistats/%s.json' %(sha256), 'w') as outfile:
                json.dump(apistats_dict, outfile)

            #if line_num == 5:
                #break
