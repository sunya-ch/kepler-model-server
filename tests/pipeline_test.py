import os
import sys

#################################################################
# import internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from train import NewPipeline
from util import get_valid_feature_group_from_queries, PowerSourceMap
from util.loader import DEFAULT_PIPELINE

from prom_test import get_query_results, prom_output_path, prom_output_filename
from extractor_test import test_extractors, test_energy_source
from isolator_test import test_isolators, test_profiles
from trainer_test import test_trainer_names, assert_train

def assert_pipeline(pipeline, query_results, feature_group, energy_source, energy_components):
    success, abs_data, dyn_data = pipeline.process(query_results, energy_components, feature_group.name, energy_source)
    assert success, "failed to process pipeline {}".format(pipeline.name) 
    for trainer in pipeline.trainers:
        if trainer.feature_group == feature_group and trainer.energy_source == energy_source:
            if trainer.node_level:
                assert_train(trainer, abs_data, energy_components)
            else:
                assert_train(trainer, dyn_data, energy_components)

def process(save_pipeline_name=DEFAULT_PIPELINE, prom_save_path=prom_output_path, prom_save_name=prom_output_filename, abs_trainer_names=test_trainer_names, dyn_trainer_names=test_trainer_names, extractors=test_extractors, isolators=test_isolators, target_energy_sources=[test_energy_source]):
    query_results = get_query_results(save_path=prom_save_path, save_name=prom_save_name)
    valid_feature_groups = get_valid_feature_group_from_queries(query_results.keys()) 
    for extractor in extractors:
        for isolator in isolators:
            energy_components = PowerSourceMap[test_energy_source]
            pipeline = NewPipeline(save_pipeline_name, test_profiles, abs_trainer_names, dyn_trainer_names, extractor=extractor, isolator=isolator, target_energy_sources=target_energy_sources ,valid_feature_groups=valid_feature_groups)
            for feature_group in valid_feature_groups:
                assert_pipeline(pipeline, query_results, feature_group, test_energy_source, energy_components)

if __name__ == '__main__':
    process()