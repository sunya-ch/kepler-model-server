############################################################
##
## profile_background
## generate a profile from prom query
## 
## ./python profile_background.py query_output_folder
## e.g., ./python profile_background.py ../tests/prom_output
##
## input must be a query output of idle state
##
## profile saved in data/profile
##  [source].json
##   {component: {node_type: {min_watt: ,max_watt: } }}
############################################################

import sys
import os

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

from sklearn.preprocessing import StandardScaler

from train import DefaultExtractor, node_info_column, component_to_col

import pandas as pd
import json

extractor = DefaultExtractor()

# TO-DO: add other source
energy_source_component_map = {
    "rapl": ["package", "core", "uncore", "dram"]
}
min_watt_key = "min_watt"
max_watt_key = "max_watt"

profile_path = scaler_top_path = os.path.join(os.path.dirname(__file__), 'data', 'profile')

def read_query_results(query_path):
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(query_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(query_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results

def load_profile(source):
    profile_filename = os.path.join(profile_path, source + ".json")
    if not os.path.exists(profile_filename):
        profile = dict()
        for component in energy_source_component_map[source]:
            profile[component] = dict()
    else:
        with open(profile_filename) as f:
            profile = json.load(f)
    return profile

def save_profile(profile, source):
    profile_filename = os.path.join(profile_path, source + ".json")
    with open(profile_filename, "w") as f:
        json.dump(profile, f)

def process(query_path):
    query_results = read_query_results(query_path)
    node_info_data = extractor.get_system_category(query_results)
    if node_info_data is None:
        print("No Node Info")
        return None
    node_types = pd.unique(node_info_data[node_info_column])
    for source, energy_components in energy_source_component_map.items():
        profile = load_profile(source)
        for node_type in node_types:
            power_data= extractor.get_power_data(query_results, energy_components, source)
            power_data = power_data.join(node_info_data)
            power_labels = power_data.columns
            for component in energy_components:
                power_label = component_to_col(component) 
                related_labels = [label for label in power_labels if power_label in label]
                # filter and extract features
                power_values = power_data[power_data[node_info_column]==node_type][related_labels].sum(axis=1)
                time_values = power_data.index.values
                seconds = time_values[1] - time_values[0]
                max_watt = power_values.max()/seconds
                min_watt = power_values.min()/seconds
                node_type_key = str(node_type)
                if node_type_key not in profile[component]:
                    profile[component][node_type_key] = {
                        min_watt_key: min_watt,
                        max_watt_key: max_watt
                    }
                else:
                    if min_watt < profile[component][node_type_key][min_watt_key]:
                        profile[component][node_type_key][min_watt_key] = min_watt
                    if max_watt > profile[component][node_type_key][max_watt_key]:
                        profile[component][node_type_key][max_watt_key] = max_watt
        print(profile)
        save_profile(profile, source)
        
if __name__ == "__main__":
    query_path = sys.argv[1]
    process(query_path)
