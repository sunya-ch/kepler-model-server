import os
import sys

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from pipeline_test import process

from util import PowerSourceMap, FeatureGroup

trainer_names = [ 'GradientBoostingRegressorTrainer', 'SGDRegressorTrainer', 'XgboostFitTrainer' ]
valid_feature_groups = [ FeatureGroup.BPFOnly, FeatureGroup.CgroupOnly ]

if __name__ == '__main__':
    process(target_energy_sources=PowerSourceMap.keys(), abs_trainer_names=trainer_names, dyn_trainer_names=trainer_names, valid_feature_groups=valid_feature_groups)