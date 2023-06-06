from sklearn.neighbors import KNeighborsRegressor

import os
import sys
trainer_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(trainer_path)

from trainer.scikit import ScikitTrainer

model_class = "scikit"

class KNeighborsRegressorTrainer(ScikitTrainer):
    def __init__(self, profiles, energy_components, feature_group, energy_source, node_level, pipeline_name):
        super(KNeighborsRegressorTrainer, self).__init__(profiles, energy_components, feature_group, energy_source, node_level, pipeline_name=pipeline_name)
        self.fe_files = []
    
    def init_model(self):
        return KNeighborsRegressor(n_neighbors=6)