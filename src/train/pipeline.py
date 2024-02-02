import os
import sys
import threading
import pandas as pd

cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)

from profiler.node_type_index import NodeTypeIndexCollection
from extractor import DefaultExtractor
from isolator import MinIdleIsolator

from train_types import PowerSourceMap, FeatureGroups, ModelOutputType
from prom_types import node_info_column
from config import model_toppath, ERROR_KEY
from loader import get_all_metadata, get_pipeline_path, get_metadata_df, get_archived_file
from saver import save_pipeline_metadata

from format import print_bounded_multiline_message, time_to_str

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

import shutil
import datetime

def load_class(module_name, class_name):
    path = os.path.join(os.path.dirname(__file__), '{}/{}'.format(module_name, class_name))
    sys.path.append(path)
    import importlib
    module_path = importlib.import_module('train.{}.{}.main'.format(module_name, class_name))
    return getattr(module_path, class_name)

def run_train(trainer, data, power_labels, pipeline_lock):
    trainer.process(data, power_labels, pipeline_lock=pipeline_lock)

class Pipeline():
    def __init__(self, name, trainers, extractor, isolator):
        self.extractor = extractor
        self.isolator = isolator
        self.trainers = trainers
        self.name = name
        self.lock = threading.Lock()
        self.path = get_pipeline_path(model_toppath=model_toppath ,pipeline_name=self.name)
        self.node_collection = NodeTypeIndexCollection(self.path)
        self.metadata = dict()
        self.metadata["name"] = self.name
        self.metadata["isolator"] = isolator.get_name()
        self.metadata["extractor"] = extractor.get_name()
        self.metadata["abs_trainers"] = [trainer.__class__.__name__ for trainer in trainers if trainer.node_level]
        self.metadata["dyn_trainers"] = [trainer.__class__.__name__ for trainer in trainers if not trainer.node_level]
        self.metadata["init_time"] = time_to_str(datetime.datetime.utcnow())

    def get_abs_data(self, query_results, energy_components, feature_group, energy_source, aggr):
        extracted_data, power_labels, _, _ = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=True, aggr=aggr)
        return extracted_data, power_labels
    
    def get_dyn_data(self, query_results, energy_components, feature_group, energy_source):
        extracted_data, power_labels, _, _ = self.extractor.extract(query_results, energy_components, feature_group, energy_source, node_level=False)
        if extracted_data is None or power_labels is None:
            return None
        isolated_data = self.isolator.isolate(extracted_data, label_cols=power_labels, energy_source=energy_source)
        return isolated_data
    
    def prepare_data(self, input_query_results, energy_components, energy_source, feature_group, aggr=True):
        query_results = input_query_results.copy()
        # 1. get abs_data
        extracted_data, power_labels = self.get_abs_data(query_results, energy_components, feature_group, energy_source, aggr=aggr)
        if extracted_data is None:
            self.print_log("cannot extract data")
            return None, None, None
        self.print_log("{} extraction done.".format(feature_group))
        abs_data = extracted_data.copy()
        # 2. get dyn_data
        isolated_data  = self.get_dyn_data(query_results, energy_components, feature_group, energy_source)
        if isolated_data is None:
            self.print_log("cannot isolate data")
            return abs_data, None, power_labels
        self.print_log("{} isolation done.".format(feature_group))
        dyn_data = isolated_data.copy()
        return abs_data, dyn_data, power_labels
    
    def prepare_data_from_input_list(self, input_query_results_list, energy_components, energy_source, feature_group, aggr=True):
        index = 0
        abs_data_list = []
        dyn_data_list = []
        power_labels = None
        for input_query_results in input_query_results_list:
            extracted_data, isolated_data, extracted_labels = self.prepare_data(input_query_results, energy_components, energy_source, feature_group, aggr)
            if extracted_data is None: 
                self.print_log("cannot extract data index={}".format(index))
                continue
            abs_data_list += [extracted_data]
            if power_labels is None:
                # set power_labels once
                power_labels = extracted_labels
            if isolated_data is None:
                self.print_log("cannot isolate data index={}".format(index))
                continue
            dyn_data_list += [isolated_data]
            index += 1
        if len(abs_data_list) == 0:
            self.print_log("cannot get abs data")
            return None, None, None
        abs_data = pd.concat(abs_data_list)
        if len(dyn_data_list) == 0:
            self.print_log("cannot get dyn data")
            return abs_data, None, power_labels
        dyn_data = pd.concat(dyn_data_list)
        return abs_data, dyn_data, power_labels

    def _train(self, abs_data, dyn_data, power_labels, energy_source, feature_group):
        # start the thread pool
        with ThreadPoolExecutor(2) as executor:
            futures = []
            for trainer in self.trainers:
                if trainer.feature_group_name != feature_group:
                    print("Skip feature: ", feature_group)
                    continue
                if trainer.energy_source != energy_source:
                    print("Skip energy source: ", energy_source)
                    continue
                if trainer.node_level and abs_data is not None:
                    future = executor.submit(run_train, trainer, abs_data, power_labels, pipeline_lock=self.lock)
                    futures += [future]
                elif dyn_data is not None:
                    future = executor.submit(run_train, trainer, dyn_data, power_labels, pipeline_lock=self.lock)
                    futures += [future]
            self.print_log('Waiting for {} trainers to complete...'.format(len(futures)))
            wait(futures)
            self.print_log('{}/{} trainers are trained from {} to {}'.format(len(futures), len(self.trainers), feature_group, energy_source))
            

    def process(self, input_query_results, energy_components, energy_source, feature_group, aggr=True, replace_node_type=None):
        self.print_log("{} start processing.".format(feature_group))
        abs_data, dyn_data, power_labels = self.prepare_data(input_query_results, energy_components, energy_source, feature_group, aggr)
        if abs_data is None and dyn_data is None:
            return False, None, None
        if replace_node_type is not None:
            self.print_log("Replace Node Type:  {}".format(replace_node_type))
            abs_data[node_info_column] = replace_node_type
            dyn_data[node_info_column] = replace_node_type
        self._train(abs_data, dyn_data, power_labels, energy_source, feature_group)
        self.print_pipeline_process_end(energy_source, feature_group, abs_data, dyn_data)
        self.metadata["last_update_time"] = time_to_str(datetime.datetime.utcnow())
        return True, abs_data, dyn_data
    
    def process_multiple_query(self, input_query_results_list, energy_components, energy_source, feature_group, aggr=True, replace_node_type=None):
        abs_data, dyn_data, power_labels = self.prepare_data_from_input_list(input_query_results_list, energy_components, energy_source, feature_group, aggr)
        if (abs_data is None or len(abs_data) == 0) and (dyn_data is None or len(dyn_data) == 0):
            return False, None, None
        if replace_node_type is not None:
            self.print_log("Replace Node Type: {}".format(replace_node_type))
            abs_data[node_info_column] = replace_node_type
            dyn_data[node_info_column] = replace_node_type
        self._train(abs_data, dyn_data, power_labels, energy_source, feature_group)
        self.print_pipeline_process_end(energy_source, feature_group, abs_data, dyn_data)
        self.metadata["last_update_time"] = time_to_str(datetime.datetime.utcnow())
        return True, abs_data, dyn_data

    def print_log(self, message):
        print("{} pipeline: {}".format(self.name, message), flush=True)

    def save_metadata(self):
        all_metadata = get_all_metadata(model_toppath, self.name)
        for energy_source, model_type_metadata in all_metadata.items():
            for model_type, metadata_df in model_type_metadata.items():
                metadata_df = metadata_df.sort_values(by=["feature_group", ERROR_KEY])
                save_pipeline_metadata(self.path, self.metadata, energy_source, model_type, metadata_df)
    
    def print_pipeline_process_end(self, energy_source, feature_group, abs_data, dyn_data):
        abs_messages = []
        dyn_messages = []
        if abs_data is not None:
            abs_trainer_names = set([trainer.__class__.__name__ for trainer in self.trainers if trainer.node_level])
            abs_metadata_df, abs_group_path = get_metadata_df(model_toppath, ModelOutputType.AbsPower.name, feature_group, energy_source, self.name)
            node_types = pd.unique(abs_metadata_df[node_info_column])
            
            abs_messages = [
                "Pipeline {} has finished for modeling {} power by {} feature".format(self.name, energy_source, feature_group),
                "    Extractor: {}".format(self.metadata["extractor"]),
                "    Isolator: {}".format(self.metadata["isolator"]),
                "Absolute Power Modeling:",
                "    Input data size: {}".format(len(abs_data)),
                "    Model Trainers: {}".format(abs_trainer_names),
                "    Output: {}".format(abs_group_path),
                " "
            ]
            for node_type in node_types:
                filtered_data = abs_metadata_df[abs_metadata_df[node_info_column]==node_type]
                min_mae = -1 if len(filtered_data) == 0 else filtered_data.loc[filtered_data[ERROR_KEY].idxmin()][ERROR_KEY]
                abs_messages += [ "    NodeType {} Min {}: {}".format(node_type, ERROR_KEY, min_mae) ]
            abs_messages += [" "]

        if dyn_data is not None:
            dyn_trainer_names = set([trainer.__class__.__name__ for trainer in self.trainers if not trainer.node_level])
            dyn_metadata_df, dyn_group_path = get_metadata_df(model_toppath, ModelOutputType.DynPower.name, feature_group, energy_source, self.name)
            dyn_messages = [
                "Dynamic Power Modeling:",
                "    Input data size: {}".format(len(dyn_data)),
                "    Model Trainers: {}".format(dyn_trainer_names),
                "    Output: {}".format(dyn_group_path),
            ]
            for node_type in node_types:
                filtered_data = dyn_metadata_df[dyn_metadata_df[node_info_column]==node_type]
                min_mae = -1 if len(filtered_data) == 0 else filtered_data.loc[filtered_data[ERROR_KEY].idxmin()][ERROR_KEY]
                dyn_messages += [ "    NodeType {} Min {}: {}".format(node_type, ERROR_KEY, min_mae) ]

        messages = abs_messages + dyn_messages
        print_bounded_multiline_message(messages)

    def archive_pipeline(self):
        save_path = os.path.join(model_toppath, self.name)
        archived_file = get_archived_file(model_toppath, self.name) 
        self.print_log("archive pipeline :" + archived_file)
        self.print_log("save_path :" + save_path)
        shutil.make_archive(save_path, 'zip', save_path)

def initial_trainers(trainer_names, node_level, pipeline_name, target_energy_sources, valid_feature_groups):
    trainers = []
    for energy_source in target_energy_sources:
        energy_components = PowerSourceMap[energy_source]
        for feature_group in valid_feature_groups:
            for trainer_name in trainer_names:
                trainer_class = load_class("trainer", trainer_name)
                try:
                    trainer_class = load_class("trainer", trainer_name)
                except Exception as e:
                    print("failed to load trainer ", trainer_name, e)
                    continue
                trainer = trainer_class(energy_components, feature_group.name, energy_source, node_level, pipeline_name=pipeline_name)
                trainers += [trainer]
    return trainers

def NewPipeline(pipeline_name, abs_trainer_names, dyn_trainer_names, extractor=DefaultExtractor(), isolator=MinIdleIsolator(), target_energy_sources=PowerSourceMap.keys(), valid_feature_groups=FeatureGroups.keys()):
    abs_trainers = initial_trainers(abs_trainer_names, node_level=True, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    dyn_trainers = initial_trainers(dyn_trainer_names, node_level=False, pipeline_name=pipeline_name, target_energy_sources=target_energy_sources, valid_feature_groups=valid_feature_groups)
    trainers = abs_trainers + dyn_trainers
    return Pipeline(pipeline_name, trainers, extractor, isolator)
    
