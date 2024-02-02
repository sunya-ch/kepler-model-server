import os
import sys

import datetime

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

cur_path = os.path.join(os.path.dirname(__file__))
sys.path.append(cur_path)

from validator import get_validated_export_items, BestModelCollection
from loader import load_metadata, load_node_type_index, get_version_path, get_export_path
from format import time_to_str
from writer import generate_pipeline_page, generate_report_results, generate_pipeline_readme, append_version_readme, get_workload_content
from config import ERROR_KEY

repo_url = "https://raw.githubusercontent.com/sustainable-computing-io/kepler-model-db/main/models"

def export(data_path, pipeline_path, db_path, publisher, collect_date, inputs):
    # load pipeline metadata
    pipeline_metadata = load_metadata(pipeline_path)
    if pipeline_metadata is None:
        print("no pipeline metadata")
        return
    # add publish information to pipeline metadata
    pipeline_metadata["publisher"] = publisher
    pipeline_metadata["collect_time"] = time_to_str(collect_date)
    pipeline_metadata["export_time"] = time_to_str(datetime.datetime.utcnow())

    node_type_index_json = load_node_type_index(pipeline_path)
    if node_type_index_json is None:
        print("no node type index")
        return
    node_types = node_type_index_json.keys()
    best_model_collections = dict()
    for node_type in node_types:
        best_model_collections[int(node_type)] = BestModelCollection(ERROR_KEY)
    
    # get path
    pipeline_name = pipeline_metadata["name"]
    local_export_path = get_export_path(db_path, pipeline_name)
    local_version_path = get_version_path(db_path)
    remote_version_path = get_version_path(repo_url, assure=False)
   
    # get validated export items (models)
    export_items = get_validated_export_items(pipeline_path, pipeline_name)
    
    for export_item in export_items:
        # export models
        export_item.export(local_version_path)
        # update best model
        best_model_collections[export_item.node_type].compare_new_item(export_item)
    
    # generate pipeline page
    workload_content = workload_content = get_workload_content(data_path, inputs)
    generate_pipeline_page(local_version_path, pipeline_metadata, workload_content)
    # generate error report page
    generate_report_results(local_export_path, best_model_collections, node_type_index_json, remote_version_path)
    # generate validation result page
    generate_pipeline_readme(pipeline_name, local_export_path, node_type_index_json, best_model_collections)
    # add new pipeline item to version path
    append_version_readme(local_version_path, pipeline_metadata)