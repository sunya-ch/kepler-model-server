# extractor_test.py
#   call 

import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

from train import MinIdleIsolator

from extractor_test import extractor_output_path, expected_power_columns

test_isolators = [MinIdleIsolator()]

import pandas as pd

target_suffix = "_False.csv"

def read_extractor_results():
    results = dict()
    isolate_target_filenames = [ filename for filename in os.listdir(extractor_output_path) if filename[len(filename)-len(target_suffix):] == "_False.csv" ]
    for filename in isolate_target_filenames:
        extractor_name = filename[0:len(filename)-len(target_suffix)] # remove "_False.csv"
        filepath = os.path.join(extractor_output_path, filename)
        results[extractor_name] = pd.read_csv(filepath)
    return results

def assert_isolate(extractor_result, isolated_data):
    isolated_data_column_names = isolated_data.columns
    assert isolated_data is not None, "isolated data is None"
    assert len(extractor_result) == len(isolated_data), "unexpected column length: expected {}, got {}({}) ".format(len(extractor_result), isolated_data_column_names, len(isolated_data_column_names))
        
if __name__ == '__main__':
    extractor_results = read_extractor_results()
    for test_instance in test_isolators:
        for extractor_name, extractor_result in extractor_results.items():
            isolated_data = test_instance.isolate(extractor_result, label_cols=expected_power_columns)
            assert_isolate(extractor_result, isolated_data)