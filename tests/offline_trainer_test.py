# offline_trainer_test.py
#
# client (test):
#   python tests/offline_trainer_test.py dataset_name <src_json_file> <idle_filepath> <save_path>
# output will be saved at
# save_path |- dataset_name |- AbsPower |- energy_source |- feature_group |- metadata.json, ...
#                           |- DynPower |- energy_source |- feature_group |- metadata.json, ...
# test offline trainer
#

import requests

import os
import sys
import shutil
import json
import codecs

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from train.offline_trainer import TrainAttribute, TrainRequest, serve_port


from model_server_test import TMP_FILE
from util import get_valid_feature_group_from_queries, query_result_to_dict
from prom_test import get_query_results
from extractor_test import test_energy_source

from train.prom.prom_query import prom_responses_to_results
from util.loader import get_pipeline_path, DEFAULT_PIPELINE

offline_trainer_output_path = os.path.join(os.path.dirname(__file__), 'data', 'offline_trainer_output')

if not os.path.exists(offline_trainer_output_path):
    os.mkdir(offline_trainer_output_path)

target_suffix = "_True.csv"

abs_trainer_names = ['GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer', 'PolynomialRegressionTrainer', 'SVRRegressorTrainer']
dyn_trainer_names = ['GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'KNeighborsRegressorTrainer', 'LinearRegressionTrainer', 'PolynomialRegressionTrainer', 'SVRRegressorTrainer']

energy_sources = ['rapl']
feature_groups = ['CounterOnly', 'CgroupOnly', 'BPFOnly', 'KubeletOnly']

profiles = dict()

# requested isolators
isolators = {
    "MinIdleIsolator": {},
    "NoneIsolator": {},
    "ProfileBackgroundIsolator": {},
    "TrainIsolator": {"abs_pipeline_name": DEFAULT_PIPELINE}
}

def get_target_path(save_path, energy_source, feature_group):
    power_path = os.path.join(save_path, energy_source)
    if not os.path.exists(power_path):
        os.mkdir(power_path)
    feature_path = os.path.join(save_path, feature_group)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    return feature_path

import json

def make_request(dataset_name, idle_data, isolator, isolator_args, data, feature_group,  energy_source, save_path):
    idle_data_dict = query_result_to_dict(idle_data)
    data_dict = query_result_to_dict(data)
    trainer = TrainAttribute(abs_trainer_names, dyn_trainer_names, idle_data_dict, isolator, isolator_args)
    request = TrainRequest(dataset_name, trainer=None, data_dict=data_dict, feature_group=feature_group, energy_source=energy_source)
    request.trainer = json.dumps(trainer.__dict__)
    print(request)
    request_json = json.dumps(request.__dict__)
    # send request
    response = requests.post('http://localhost:{}/train'.format(serve_port), json=request_json)
    assert response.status_code == 200, response.text
    with codecs.open(TMP_FILE, 'wb') as f:
        f.write(response.content)
    # unpack response
    shutil.unpack_archive(TMP_FILE, save_path)
    os.remove(TMP_FILE)

def get_pipeline_name(dataset_name, isolator):
    return "{}_{}".format(dataset_name, isolator)  

def process(dataset_name, train_data, idle_data, feature_group, energy_source=test_energy_source, isolators=isolators, target_path=offline_trainer_output_path):
    for isolator, isolator_args in isolators.items():
        print("Isolator: ", isolator)
        pipeline_name = get_pipeline_name(dataset_name, isolator)
        save_path = os.path.join(target_path, pipeline_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        make_request(pipeline_name, idle_data, isolator, isolator_args, train_data, feature_group, energy_source, save_path)
    
if __name__ == '__main__':
    dataset_name = "sample_data"
    idle_data = get_query_results(save_name="idle")
    train_data = get_query_results()
    valid_feature_groups = get_valid_feature_group_from_queries(train_data.keys())
    for fg in valid_feature_groups:
        feature_group = fg.name  
        process(dataset_name, train_data, idle_data, feature_group)
    