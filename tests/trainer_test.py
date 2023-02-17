import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '../server')
train_path = os.path.join(os.path.dirname(__file__), '../server/train')

sys.path.append(server_path)
sys.path.append(train_path)

from train import load_class, load_all_profiles
from isolator_test import isolator_output_path
from extractor_test import extractor_output_path, energy_components, energy_source, feature_group, expected_power_columns, node_info_column

import pandas as pd

target_suffix = "_False.csv"

trainer_names = ['GradientBoostingRegressorTrainer']

def read_extractor_results():
    results = dict()
    target_filenames = [ filename for filename in os.listdir(extractor_output_path) if filename[len(filename)-len(target_suffix):] == "_True.csv" ]
    for filename in target_filenames:
        extractor_name = filename[0:len(filename)-len(target_suffix)] # remove "_True.csv"
        filepath = os.path.join(extractor_output_path, filename)
        results[extractor_name] = pd.read_csv(filepath)
    return results

def read_isolator_results():
    results = dict()
    target_filenames = [ filename for filename in os.listdir(isolator_output_path)]
    for filename in target_filenames:
        isolator_name = filename.split("_")[0]
        filepath = os.path.join(isolator_output_path, filename)
        results[isolator_name] = pd.read_csv(filepath)
    return results

def assert_train(trainer, data):
    node_types = pd.unique(data[node_info_column])
    for node_type in node_types:
        node_type_filtered_data = data[data[node_info_column] == node_type]
        X_values = node_type_filtered_data[trainer.features].values
        for component in energy_components:
            output = trainer.predict(node_type, component, X_values)
            assert len(output) == len(X_values), "length of predicted values != features ({}!={})".format(len(output), len(X_values))

if __name__ == '__main__':
    profiles = load_all_profiles()
    extractor_results = read_extractor_results()
    isolated_results = read_isolator_results()
    for trainer_name in trainer_names:
        trainer_class = load_class("trainer", trainer_name)
        node_level = True
        trainer = trainer_class(profiles, energy_components, feature_group, energy_source, node_level)
        for result in extractor_results.values():
            trainer.process(result, expected_power_columns)
            assert_train(trainer, result)
        node_level = False
        trainer = trainer_class(profiles, energy_components, feature_group, energy_source, node_level)
        for result in isolated_results.values():
            trainer.process(result, expected_power_columns)
            assert_train(trainer, result)