## Test Server API (model selection)

import os
import sys
import shutil
import requests
import codecs
import json

util_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'util')
server_path = os.path.join(os.path.dirname(__file__), '../src/server')
estimate_path = os.path.join(os.path.dirname(__file__), '../src/estimate')
sys.path.append(util_path)
sys.path.append(server_path)
sys.path.append(estimate_path)

from train_types import FeatureGroups, FeatureGroup, ModelOutputType
from loader import base_model_url, load_remote_json, load_metadata
from saver import NODE_TYPE_INDEX_FILENAME
from config import download_path, default_pipelines
from model_server_test import get_model_request_json
from model_server_connector import list_all_models
from model_server import MODEL_SERVER_PORT

TMP_FILE = 'download.zip'

# set environment
os.environ['MODEL_SERVER_URL'] = 'http://localhost:8100'

def get_node_types(energy_source):
    pipeline_name = default_pipelines[energy_source]
    url_path = os.path.join(base_model_url, pipeline_name, NODE_TYPE_INDEX_FILENAME)
    return load_remote_json(url_path)

def make_request_with_spec(metrics, output_type, node_type=-1, weight=False, trainer_name="", energy_source='rapl-sysfs', spec=None):
    model_request = get_model_request_json(metrics, output_type, node_type, weight, trainer_name, energy_source)
    model_request["spec"] = spec
    response = requests.post('http://localhost:{}/model'.format(MODEL_SERVER_PORT), json=model_request)
    assert response.status_code == 200, response.text
    if weight:
        weight_dict = json.loads(response.text)
        assert len(weight_dict) > 0, "weight dict must contain one or more than one component"
        for weight_values in weight_dict.values():
            weight_length = len(weight_values['All_Weights']['Numerical_Variables'])
            expected_length = len(metrics)
            assert weight_length <= expected_length, "weight metrics should covered by the requested {} > {}".format(weight_length, expected_length)
        return weight_dict["model_name"], weight_length.keys()
    else:
        output_path = os.path.join(download_path, output_type.name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        with codecs.open(TMP_FILE, 'wb') as f:
            f.write(response.content)
        shutil.unpack_archive(TMP_FILE, output_path)
        metadata = load_metadata(output_path)
        os.remove(TMP_FILE)
        return metadata["model_name"], metadata["features"]

def check_select_model(model_name, features, best_model_map):
    assert model_name != "", "model name should not be empty."
    found = False
    for cmp_fg_name, expected_best_model_without_node_type in best_model_map.items():
        cmp_metrics = FeatureGroups[FeatureGroup[cmp_fg_name]]
        if cmp_metrics == features:
            found = True
            assert model_name == expected_best_model_without_node_type, f"should select best model {expected_best_model_without_node_type} (select {model_name}) - {output_type}/{fg_name}"
            break
    assert found, f"must found matched best model without node_type for {features}: {best_model_map}"

def process(node_type, info, output_type, energy_source, valid_fgs, best_model_by_source):
    expected_suffix = f"_{node_type}"
    for fg_name in valid_fgs.keys():
        metrics = FeatureGroups[FeatureGroup[fg_name]]
        model_name, features = make_request_with_spec(metrics, output_type, energy_source=energy_source)
        check_select_model(model_name, features, best_model_by_source)
        model_name, features = make_request_with_spec(metrics, output_type, node_type=node_type, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        model_name, features = make_request_with_spec(metrics, output_type, spec=info['attrs'], energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        fixed_some_spec = {'processor': info['attrs']['processor'], 'memory': info['attrs']['memory']}
        model_name, features = make_request_with_spec(metrics, output_type, spec=fixed_some_spec, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)
        uncovered_spec = info['attrs'].copy()
        uncovered_spec['processor'] = "_".join(uncovered_spec['processor'].split("_")[:-1])
        model_name, features = make_request_with_spec(metrics, output_type, spec=uncovered_spec, energy_source=energy_source)
        assert expected_suffix in model_name, "model must be a matched type"
        check_select_model(model_name, features, valid_fgs)

test_energy_sources = ["rapl-sysfs"]

if __name__ == '__main__':
    # test getting model from server
    os.environ['MODEL_SERVER_ENABLE'] = "true"
    available_models = list_all_models()
    assert len(available_models) > 0, "must have more than one available models"
    print("Available Models:", available_models)
    for energy_source in test_energy_sources:
        node_types = get_node_types(energy_source)
        best_model_by_source_map = list_all_models(energy_source=energy_source)
        for node_type, info in node_types.items():
            available_models = list_all_models(node_type=node_type, energy_source=energy_source)
            if len(available_models) > 0:
                for output_type_name, valid_fgs in available_models.items():
                    output_type = ModelOutputType[output_type_name]
                    process(node_type, info, output_type, energy_source, valid_fgs, best_model_by_source=best_model_by_source_map[output_type_name])
            else:
                print(f"skip {energy_source}/{node_type} because on available models")
