## TO-DO: change to train for absolute power not components
import os
import sys

train_path = os.path.join(os.path.dirname(__file__), '../../')
prom_path = os.path.join(os.path.dirname(__file__), '../../../prom')
sys.path.append(train_path)
sys.path.append(prom_path)

from pipeline import TrainPipeline
from train_types import FeatureGroup, FeatureGroups, CORE_COMPONENT, DRAM_COMPONENT, ModelOutputType
from pipe_util import train_model_given_data_and_type, create_prometheus_core_dataset, create_prometheus_dram_dataset
from pipe_util import generate_core_regression_model, generate_dram_regression_model, coeff_determination
from pipe_util import dram_model_labels, core_model_labels
from pipe_util import merge_model
from transformer import KerasFullPipelineFeatureTransformer
from keras.models import load_model
from keras import backend as K
import pickle


from prom.prom_query import NODE_STAT_QUERY

MODEL_CLASS = 'keras'
FE_FILE = 'merge.pkl'

class KerasFullPipeline(TrainPipeline):
    def __init__(self):
        self.mse = None
        self.mse_val = None
        self.mae = None
        self.mae_val = None
        model_name = KerasFullPipeline.__name__
        model_file = model_name

        super(KerasFullPipeline, self).__init__(model_name, MODEL_CLASS, model_file, FeatureGroups[FeatureGroup.Full], ModelOutputType.AbsPower)
        self.fe = KerasFullPipelineFeatureTransformer(self.features, dram_model_labels, core_model_labels)
        fe_file_path = os.path.join(self.save_path, FE_FILE)
        with open(fe_file_path, 'wb') as f:
            pickle.dump(self.fe, f)
            self.fe_files = [FE_FILE]

    def train(self, prom_client):
        node_stat_data = prom_client.get_data(NODE_STAT_QUERY, None)
        if node_stat_data is None or int(node_stat_data['energy_in_pkg_joule'].sum()) == 0:
            # cannot train with package-level info
            return
        core_train, core_val, core_test = create_prometheus_core_dataset(node_stat_data['cpu_architecture'], node_stat_data['cpu_cycles'], node_stat_data['cpu_instr'], node_stat_data['cpu_time'], node_stat_data['energy_in_core_joule'])
        dram_train, dram_val, dram_test = create_prometheus_dram_dataset(node_stat_data['cpu_architecture'], node_stat_data['cache_miss'], node_stat_data['container_memory_working_set_bytes'], node_stat_data['energy_in_dram_joule'])
        core_model = self.load_model(core_train, CORE_COMPONENT)
        dram_model = self.load_model(dram_train, DRAM_COMPONENT)
        core_model, _, _, core_mae_metric = train_model_given_data_and_type(core_model, core_train, core_val, core_test, CORE_COMPONENT)
        dram_model, _, _, dram_mae_metric = train_model_given_data_and_type(dram_model, dram_train, dram_val, dram_test, DRAM_COMPONENT)
        core_model.save(self._get_model_path(CORE_COMPONENT), save_format='tf', include_optimizer=True)
        dram_model.save(self._get_model_path(DRAM_COMPONENT), save_format='tf', include_optimizer=True)
        merged_model = merge_model(core_model, dram_model)
        merged_model.save(self._get_model_path(self.model_file), save_format='tf', include_optimizer=True)
        metadata = dict()
        metadata['mae'] = core_mae_metric + dram_mae_metric
        self.update_metadata(metadata)

    def _get_model_path(self, model_type):
        return os.path.join(self.save_path, model_type)

    def load_model(self, train_dataset, model_type):
        filepath = self._get_model_path(model_type)
        model_exists = os.path.exists(filepath)
        if model_exists:
            new_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
        elif model_type == CORE_COMPONENT:
            new_model = generate_core_regression_model(train_dataset)
        elif model_type == DRAM_COMPONENT:
            new_model = generate_dram_regression_model(train_dataset)
        else:
            "wrong type: " + model_type
        return new_model

    def predict(self, data):
        x_values = data[self.features].values
        print(x_values.shape)
        fe_filepath = os.path.join(self.save_path, FE_FILE)
        fe_item = pickle.load(open(fe_filepath, 'rb'))
        transformed_values = fe_item.transform(x_values)
        model_filepath = os.path.join(self.save_path, self.model_file)
        merged_model = load_model(model_filepath, custom_objects={'coeff_determination': coeff_determination})
        result = merged_model.predict(transformed_values)
        print(result)