import sys
import os
server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from extractor.extractor import DefaultExtractor, node_info_column, component_to_col, TIMESTAMP_COL, UNKNOWN_NODE_INFO
from isolator.isolator import MinIdleIsolator
from train_types import FeatureGroups, FeatureGroup, get_feature_group, is_weight_output