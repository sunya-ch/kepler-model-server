import os
import sys
import pandas as pd

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from abc import ABCMeta, abstractmethod
from extractor.extractor import container_id_colname, TIMESTAMP_COL

container_indexes = [TIMESTAMP_COL, container_id_colname]

class Isolator(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def isolate(self, data, profile=None):
        return NotImplemented

def exclude_target_container_usage(data, target_container_id):
    target_container_data = data[data[container_id_colname]==target_container_id]
    filled_target_container_data = data[target_container_data.columns].join(target_container_data, lsuffix='_target').fillna(0)
    filled_target_container_data.drop(columns=[col for col in filled_target_container_data.columns if '_target' in col], inplace=True)
    conditional_data = data - filled_target_container_data
    return target_container_data, conditional_data

class MinIdleIsolator(Isolator):
    def isolate(self, data, label_cols, *args):
        isolated_data = data.copy()
        for label_col in label_cols:
            min = data[label_col].min()
            isolated_data[label_col] = data[label_col] - min
        return isolated_data

class ProfileBackgroundIsolator(Isolator):
    def isolate(self, data, label_cols, *args):
        target_container_id = args[0]
        indexed_data = data.set_index(container_indexes)
        isolated_data, _ = exclude_target_container_usage(indexed_data, target_container_id)
        profiles = args[0:len(label_cols)]
        for i in range(0, len(label_cols)):
            isolated_data[label_cols[i]] -= profiles[i] 
        return isolated_data