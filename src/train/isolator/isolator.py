import os
import sys
import pandas as pd
from abc import ABCMeta, abstractmethod

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

estimate_path = os.path.join(os.path.dirname(__file__), '..', '..', 'estimate')
sys.path.append(estimate_path)


from util import PowerSourceMap
from util.extract_types import container_id_colname, col_to_component, get_num_of_unit, all_container_key
from util.prom_types import TIMESTAMP_COL, node_info_column, get_container_name_from_id

from estimate import get_background_containers, get_predicted_power_colname, get_predicted_background_power_colname, get_predicted_dynamic_background_power_colname, get_label_power_colname, get_reconstructed_power_colname

container_indexes = [TIMESTAMP_COL, container_id_colname]


class Isolator(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def isolate(self, data, **args):
        return NotImplemented

    # should return reconstruct power
    @abstractmethod
    def reconstruct(self, data, **args):
        return NotImplemented


def exclude_target_container_usage(data, target_container_id):
    target_container_data = data[data[container_id_colname]==target_container_id]
    filled_target_container_data = data[target_container_data.columns].join(target_container_data, lsuffix='_target').fillna(0)
    filled_target_container_data.drop(columns=[col for col in filled_target_container_data.columns if '_target' in col], inplace=True)
    conditional_data = data - filled_target_container_data
    return target_container_data, conditional_data

# target_containers are containers that are not in idle
# idle_data in json format
def get_target_containers(data, background_containers):
    container_ids = pd.unique(data.reset_index()[container_id_colname])
    target_containers = [container_id for container_id in container_ids if get_container_name_from_id(container_id) not in background_containers]
    return target_containers, background_containers

# isolate_container
def isolate_container(extracted_data, background_containers):
    target_containers, background_containers = get_target_containers(extracted_data, background_containers)
    target_data = extracted_data[extracted_data[container_id_colname].isin(target_containers)]
    background_data = extracted_data[~extracted_data[container_id_colname].isin(target_containers)]
    return target_data, background_data


def squeeze_data(container_level_data, label_cols):
    node_level_columns = list(label_cols) + [node_info_column]
    ratio_columns = [col for col in container_level_data.columns if "ratio" in col]
    feature_columns = [col for col in container_level_data.columns if col not in node_level_columns and col not in ratio_columns and col not in container_indexes] 
    groupped_sum_data = container_level_data.groupby([TIMESTAMP_COL]).sum()[ratio_columns + feature_columns]
    groupped_sum_data['sum_ratio'] = groupped_sum_data[ratio_columns].sum(axis=1)
    for ratio_col in ratio_columns:
        groupped_sum_data[ratio_col] /= groupped_sum_data['sum_ratio']
    groupped_sum_data = groupped_sum_data.drop(columns=['sum_ratio'])
    groupped_mean_data = container_level_data.groupby([TIMESTAMP_COL]).mean()[node_level_columns]  
    squeeze_data = groupped_sum_data.join(groupped_mean_data)
    squeeze_data[container_id_colname] = all_container_key
    return squeeze_data.reset_index()

class MinIdleIsolator(Isolator):

    def isolate(self, data, label_cols, energy_source=None):
        isolated_data = squeeze_data(data, label_cols)
        for label_col in label_cols:
            min = isolated_data[label_col].min()
            isolated_data[label_col] = isolated_data[label_col] - min
        return isolated_data
    
    def reconstruct(self, extracted_data, data_with_prediction, energy_source, label_cols):
        num_of_unit = get_num_of_unit(energy_source, label_cols)
        reconstructed_data = data_with_prediction.groupby([TIMESTAMP_COL]).sum()
        for energy_component in PowerSourceMap(energy_source):
            target_cols = [col for col in label_cols if energy_component in col]
            predicted_colname = get_predicted_power_colname[energy_source]
            min = extracted_data[target_cols].min().values.min()
            background_power_colname = get_predicted_background_power_colname(energy_component)
            reconstructed_data[background_power_colname] = min * num_of_unit 
            reconstructed_data[get_reconstructed_power_colname(energy_component)] = data_with_prediction[predicted_colname] + reconstructed_data[background_power_colname] 
        return reconstructed_data
    
import numpy as np

system_process_id = "system_processes/system_processes/system"

class ProfileBackgroundIsolator(Isolator):
    def __init__(self, profiles, idle_data):
        self.idle_data = idle_data
        self.profiles = profiles
        self.background_containers = get_background_containers(self.idle_data)

    def transform_profile(self, node_type, energy_source, component):
        if int(node_type) not in self.profiles:
            return np.nan    
        return self.profiles[node_type].get_background_power(energy_source, component)

    def transform_component(self, label_col):
        return col_to_component(label_col)

    def isolate(self, data, label_cols, energy_source):
        target_data, _ = isolate_container(data, self.background_containers)
        isolated_data = target_data.copy()
        try:
            for label_col in label_cols:
                component = col_to_component(label_col)
                isolated_data['profile'] = isolated_data[node_info_column].transform(self.transform_profile, energy_source=energy_source, component=component)
                if isolated_data['profile'].isnull().values.any():
                    return None
                isolated_data[label_col] = data[label_col] - isolated_data['profile']
                isolated_data.drop(columns='profile', inplace=True)
            return isolated_data
        except Exception as e:
            print(e)
            return None
        
    def reconstruct(self, extracted_data, data_with_prediction, energy_source, label_cols):
        energy_components = PowerSourceMap[energy_source]
        num_of_unit = get_num_of_unit(energy_source, label_cols)
        reconstructed_data = data_with_prediction.groupby([TIMESTAMP_COL]).sum()
        copy_data = extracted_data.copy()
        for energy_component in energy_components:
            try:
                copy_data['profile'] = copy_data[node_info_column].transform(self.transform_profile, energy_source=energy_source, component=energy_component)
                if copy_data['profile'].isnull().values.any():
                    return None       
                predicted_colname = get_predicted_power_colname[energy_source]
                background_power_colname = get_predicted_background_power_colname(energy_component)
                reconstructed_data[background_power_colname] = (copy_data['profile'] * num_of_unit)
                reconstructed_data[get_reconstructed_power_colname(energy_component)] = data_with_prediction[predicted_colname]  + reconstructed_data[background_power_colname] 
            except Exception as e:
                print(e)
                return None
        return reconstructed_data
        

# no isolation
class NoneIsolator(Isolator):

    def isolate(self, data, label_cols, energy_source=None):
        isolated_data = squeeze_data(data, label_cols)
        return isolated_data
    
    def reconstruct(self, extracted_data, data_with_prediction, energy_source, label_cols):
        return data_with_prediction
    