import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

isolator_output_path = os.path.join(os.path.dirname(__file__), 'data', 'isolator_output')

from train import MinIdleIsolator, ProfileBackgroundIsolator
from train import load_class, load_all_profiles

from extractor_test import energy_source, extractor_output_path, expected_power_columns

test_isolators = [MinIdleIsolator(), ProfileBackgroundIsolator(load_all_profiles())]
# Add customize isolator here
customize_isolators = []
for isolator_name in customize_isolators:
    test_isolators += [load_class("isolator", isolator_name)]

import pandas as pd

target_suffix = "_False.csv"

def save_results(instance, extractor_name, isolated_data):
    filename = "{}_{}.csv".format(instance.__class__.__name__, extractor_name)
    filepath = os.path.join(isolator_output_path, filename)
    isolated_data.to_csv(filepath)


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
    assert len(extractor_result.columns) == len(isolated_data_column_names), "unexpected column length: expected {}, got {}({}) ".format(len(extractor_result.columns), isolated_data_column_names, len(isolated_data_column_names))
        
if __name__ == '__main__':
    extractor_results = read_extractor_results()
    for test_instance in test_isolators:
        for extractor_name, extractor_result in extractor_results.items():
            isolated_data = test_instance.isolate(extractor_result, energy_source, label_cols=expected_power_columns)
            assert_isolate(extractor_result, isolated_data)
            save_results(test_instance, extractor_name, isolated_data)