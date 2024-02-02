import os
import sys
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)

from estimate_common import transform_and_predict, load_model_by_pickle, load_model_by_json, is_component_model
from util.train_types import main_feature, get_valid_feature_groups

import os
import sys
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(src_path)

from util import ModelOutputType

import collections.abc

class CurveFitModel():
    def __init__(self, model_path, model_name, output_type, model_file, features, fe_files, component_init=False, feature_group=None):
        self.name = model_name
        self.features = features
        if feature_group is None:
            self.feauture_group = get_valid_feature_groups(features)[0]
        else:
            self.feauture_group = feature_group
        self.output_type = ModelOutputType[output_type]

        self.comp_type = not component_init and is_component_model(model_file)
        if self.comp_type:
            self.models = dict()
            model_info = load_model_by_json(model_path, model_file)
            for comp, model_metadata in model_info.items():
                model = CurveFitModel(model_path, self.name, self.output_type.name, model_metadata['model_file'], model_metadata['features'], model_metadata['fe_files'], component_init=True)
                feature_index = main_feature(self.feauture_group.name, comp)
                model.model.set_feature_index(feature_index)
                self.models[comp] = model
        else:
            self.model = load_model_by_pickle(model_path, model_file)
            self.fe_list = []
            for fe_filename in fe_files:
                self.fe_list += [load_model_by_pickle(model_path, fe_filename)]

    def get_power(self, request):
        if self.comp_type:
            results = dict()
            for comp, model in self.models.items():
                 y, msg = transform_and_predict(model, request)
                 if msg != "":
                    return [], msg
                 if not isinstance(y, collections.abc.Sequence):
                    y = [y]
                 results[comp] = y
            return results, msg
        else:
            return transform_and_predict(self, request)

