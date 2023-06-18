# offline trainer 
# input: response, pipeline attributes
# output: model output

# server:
#   python src/train/offline_trainer.py

# client (test):
#   python tests/offline_trainer_test.py <src_json_file> <save_path>

import os
import sys

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)
model_path = os.path.join(os.path.dirname(__file__), '..', 'estimate', 'model')
sys.path.append(model_path)
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
profiler_path = os.path.join(os.path.dirname(__file__), 'profiler')
sys.path.append(profiler_path)

from prom.prom_query import prom_responses_to_results
from loader import get_pipeline_path, DEFAULT_PIPELINE, query_result_from_dict
from train_types import FeatureGroup, PowerSourceMap
from profiler import Profiler
from extractor import DefaultExtractor

import shutil

from flask import Flask, request, make_response, send_file
serve_port = 8080

"""
TrainRequest
name - pipeline/model name
output_type - target output
trainer - attribute to construct a training pipeline
    - abs_trainers - trainer class name list for absolute power training 
    - dyn_trainers - trainer class name list for dynamic power training
    - idle_data - profile dict at idle state for profile-based idle power isolation 
    - isolator - isolator key name  
        - isolator_args - isolator arguments such as training class name to predict background power, profiled idle 
data - data to train
response - prom_response from query.py

TrainResponse
zip file of pipeline folder or error message
"""

class TrainAttribute():
    def __init__(self, abs_trainers, dyn_trainers, idle_data_dict, isolator, isolator_args):
        self.abs_trainers = abs_trainers
        self.dyn_trainers = dyn_trainers
        self.idle_data_dict = idle_data_dict
        self.isolator = isolator
        self.isolator_args = isolator_args

class TrainRequest():
    def __init__(self, name, feature_group, energy_source, trainer, data_dict):
        self.name = name
        self.feature_group = feature_group
        self.energy_source = energy_source
        self.energy_components = PowerSourceMap[self.energy_source]
        if trainer is not None:
            self.trainerer = TrainAttribute(**trainer)
        self.data_dict = data_dict

    def init_trainer(self, trainer_name, node_level):
        trainer_class = load_class("trainer", trainer_name)
        trainer = trainer_class(self.profiles, self.energy_components, self.feature_group.name, self.energy_source, node_level)
        return trainer

    def init_isolator(self):
        profiler = Profiler(extractor=DefaultExtractor())
        isolator_key = self.trainer.isolator
        isolator_args = self.trainer.isolator_args
        # TODO: if idle_data is None use profile from registry
        idle_data = query_result_from_dict(self.trainer.idle_data_dict)
        idle_profile_map = profiler.process(idle_data)
        profiles = generate_profiles(idle_profile_map)
        if isolator_key == ProfileBackgroundIsolator.__class__.name:
            isolator = ProfileBackgroundIsolator(profiles, idle_data)
        elif isolator_key == train_common.TrainIsolator.__class__.name:
            if 'abs_pipeline_name' not in isolator_args:
                # use default pipeline for absolute model training in isolation
                isolator_args['abs_pipeline_name'] = DEFAULT_PIPELINE
            isolator = train_common.TrainIsolator(idle_data, profiler=profiler, abs_pipeline_name=abs_pipeline_name)
        else:
            # default init, no args
            isolator =  getattr(sys.modules[__name__], isolator_key)()
        return isolator  
    
    def init_pipeline(self):
        isolator = self.init_isolator()
        self.pipeline = NewPipeline(self.name, self.profiles, self.abs_trainer_names, self.dyn_trainer_names, extractor=DefaultExtractor(), isolator=isolator)
    
    def get_model(self):
        pipeline = req.init_pipeline()
        fg = FeatureGroup[self.feature_group]
        # train model
        data = query_result_from_dict(self.trainer.data_dict)
        pipeline.process(data, self.energy_components,fg, self.energy_source)
        # return model
        pipeline_path = get_pipeline_path(pipeline_name=self.name)
        try:
            shutil.make_archive(pipeline_path, 'zip', pipeline_path)
            return pipeline_path + '.zip'
        except Exception as e:
            print(e)
            return None

app = Flask(__name__)

# return archive file or error
@app.route('/train', methods=['POST'])
def train():
    train_request = request.get_json()
    req = TrainRequest(**train_request)
    model = req.get_model()
    if model is None:
        return make_response("Cannot train model {}".format(req.name), 400)
    else:
        try:
            return send_file(model, as_attachment=True)
        except ValueError as err:
            return make_response("Send trained model error: {}".format(err), 400)

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=True, port=serve_port)
