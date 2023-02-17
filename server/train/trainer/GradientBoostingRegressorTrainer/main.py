from sklearn.ensemble import GradientBoostingRegressor
import joblib
from urllib.request import urlopen

import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from trainer import Trainer

model_class = "scikit"

class GradientBoostingRegressorTrainer(Trainer):
    def __init__(self, profiles, energy_components, feature_group, energy_source, node_level):
        trainer_name = self.__class__.__name__
        super(GradientBoostingRegressorTrainer, self).__init__(profiles, trainer_name, model_class, energy_components, feature_group, energy_source, node_level)
        self.fe_files = []
    
    def init_model(self):
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
 
    def train(self, node_type, component, X_values, y_values):
        model = self.node_models[node_type][component]
        model.fit(X_values, y_values)

    def save_checkpoint(self, model, filepath):
        filepath += ".pkl"
        joblib.dump(model, filepath)

    def load_local_checkpoint(self, filepath):
        filepath += ".pkl"
        try:
            loaded_model = joblib.load(filepath)
            return loaded_model, True
        except:
            return None, False

    def load_remote_checkpoint(self, url_path):
        url_path += ".pkl"
        try:        
            response = urlopen(url_path)
            loaded_model = joblib.load(response)
            return loaded_model, True
        except:
            return None, False

    def should_archive(self, node_type):
        return True

    def get_basic_metadata(self, node_type):
        return dict()

    def get_mae(self, node_type, component, X_test, y_test):
        model = self.node_models[node_type][component]
        validation_loss = model.score(X_test, y_test)
        return validation_loss

    def save_model(self, component_save_path, node_type, component):
        model = self.node_models[node_type][component]
        model_filename = self.component_model_filename(component)
        filepath = os.path.join(component_save_path, model_filename)
        self.save_checkpoint(model, filepath)

    def component_model_filename(self, component):
        return component + ".pkl"
