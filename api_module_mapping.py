import json, glob, os
from collections import Counter

def gen_api_module_mapping(num_samples):
    
    filepaths = glob.glob(os.path.join("./samples_%s" % (num_samples), '*.jsonl'))
    api_module_counts = {}

    for filepath in filepaths:
        #print(filepath)
        with open (filepath,"r") as infile:
            for line_num, line in enumerate(infile):
                jsonline = json.loads(line)

                imports = jsonline["imports"]
                #loop through all modules in imports
                for module in imports:
                    module_apis = imports[module]

                    #loop through all apis under each module
                    for api in module_apis:
                        if api not in api_module_counts:
                            #create dictionary to count number of times api belonged to which module
                            api_module_counts[api] = {}

                        module_counts_dict = api_module_counts[api]
                        if module not in module_counts_dict:
                            module_counts_dict[module] = 1
                        else:
                            module_counts_dict[module] += 1

                        #assign module_counts_dict back to api_module_counts[api]?
                        #api_module_counts[api] = module_counts_dict

    #print("api_module_counts:",api_module_counts)
    api_module_mapping = {}
    for api in api_module_counts:
        #sort module_counts by descending order
        module_counts = dict(Counter(api_module_counts[api]).most_common())
        #print("module_counts:",module_counts)
        #assign api to most common module
        most_common_module = list(module_counts.keys())[0]
        api_module_mapping[api] = most_common_module

    #print("api_module_mapping:",api_module_mapping)
    with open ("./api_module_mapping/api_module_mapping_%s.json" % (num_samples), "w") as outfile:
        json.dump(api_module_mapping, outfile)

    return api_module_mapping

gen_api_module_mapping(4096)