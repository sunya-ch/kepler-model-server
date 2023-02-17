############################################################
##
## generate_scaler
## generate a scaler for each node type from prom query
## 
## ./python generate_scaler.py query_output_folder
## e.g., ./python generate_scaler.py ../tests/prom_output
##
## input must be a query output of loaded state 
##
############################################################

import sys
import os

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

from sklearn.preprocessing import StandardScaler

from train import DefaultExtractor, node_info_column, FeatureGroups, FeatureGroup, TIMESTAMP_COL

import pandas as pd
import pickle

# TO-DO: add feature group when available on kepler
feature_groups = ['KubeletOnly']

extractor = DefaultExtractor()

scaler_top_path = os.path.join(os.path.dirname(__file__), 'data', 'scaler')

def read_query_results(query_path):
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(query_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(query_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results

def save_scaler(scaler, node_type, feature_group):
    node_type_path = os.path.join(scaler_top_path, str(node_type))
    if not os.path.exists(node_type_path):
        os.mkdir(node_type_path)
    filename = os.path.join(node_type_path, feature_group + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(scaler, f)

def process(query_path):
    query_results = read_query_results(query_path)
    node_info_data = extractor.get_system_category(query_results)
    if node_info_data is None:
        print("No Node Info")
        return None
    node_types = pd.unique(node_info_data[node_info_column])
    for node_type in node_types:
        for feature_group in feature_groups:
            features = FeatureGroups[FeatureGroup[feature_group]]
            feature_data = extractor.get_feature_data(query_results, features)
            feature_data = feature_data.groupby([TIMESTAMP_COL]).sum()[features]
            feature_data = feature_data.join(node_info_data)
            node_types = pd.unique(feature_data[node_info_column])
            # filter and extract features
            x_values = feature_data[feature_data[node_info_column]==node_type][features].values
            standard_scaler = StandardScaler()
            standard_scaler.fit(x_values)
            save_scaler(standard_scaler, node_type, feature_group)

if __name__ == "__main__":
    query_path = sys.argv[1]
    process(query_path)
