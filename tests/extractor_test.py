import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

from train import DefaultExtractor, node_info_column, component_to_col, FeatureGroups, FeatureGroup
from train import load_class

from prom_test import prom_output_path

energy_components = ["package", "core", "uncore", "dram"]
num_of_package = 4
feature_group = 'KubeletOnly'
energy_source = "rapl"

extractor_output_path = os.path.join(os.path.dirname(__file__), 'data', 'extractor_output')

if not os.path.exists(extractor_output_path):
    os.mkdir(extractor_output_path)

expected_power_columns = [component_to_col(component, "package", unit_val) for component in energy_components for unit_val in range(0,4)]

test_extractors = [DefaultExtractor()]
# Add customize extractor here
customize_extractors = []
for extractor_name in customize_extractors:
    test_extractors += [load_class("extractor", extractor_name)]

import pandas as pd

def read_sample_query_results():
    results = dict()
    metric_filenames = [ metric_filename for metric_filename in os.listdir(prom_output_path) ]
    for metric_filename in metric_filenames:
        metric = metric_filename.replace(".csv", "")
        filepath = os.path.join(prom_output_path, metric_filename)
        results[metric] = pd.read_csv(filepath)
    return results


def save_results(instance, extracted_data, node_level):
    filename = "{}_{}.csv".format(instance.__class__.__name__, node_level)
    filepath = os.path.join(extractor_output_path, filename)
    extracted_data.to_csv(filepath)

def assert_extract(extracted_data, power_columns):
    extracted_data_column_names = extracted_data.columns
    # basic assert
    assert extracted_data is not None, "extracted data is None"
    assert len(power_columns) > 0, "no power label column {}".format(extracted_data_column_names)
    assert node_info_column in extracted_data_column_names, "no {} in column {}".format(node_info_column, extracted_data_column_names)
    expected_power_column_length = len(energy_components) * num_of_package
    # detail assert
    assert len(power_columns) == expected_power_column_length, "unexpected power label columns {}".format(power_columns)
    expected_col_size = expected_power_column_length + len(FeatureGroups[FeatureGroup[feature_group]]) + 1
    assert len(extracted_data_column_names) == expected_col_size, "unexpected column length: expected {}, got {}({}) ".format(expected_col_size, extracted_data_column_names, len(extracted_data_column_names))
        
if __name__ == '__main__':
    query_results = read_sample_query_results()
    for test_instance in test_extractors:
        extracted_data, power_columns = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=True)
        assert_extract(extracted_data, power_columns)
        save_results(test_instance, extracted_data, True)
        extracted_data, power_columns = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        assert_extract(extracted_data, power_columns)
        save_results(test_instance, extracted_data, False)