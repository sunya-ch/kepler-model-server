pipeline_name = 'KerasCompFullPipeline'

from random import random

import os
import sys
prom_path = os.path.join(os.path.dirname(__file__), '../../../server/train/prom')
sys.path.append(prom_path)
train_path = os.path.join(os.path.dirname(__file__), '../../../server/train')
sys.path.append(train_path)
import importlib
pipeline_module = importlib.import_module('pipelines.{}.pipe'.format(pipeline_name))
pipeline = getattr(pipeline_module, pipeline_name)()

from prom.prom_query import NODE_STAT_QUERY

if __name__ == '__main__':
    from prom.prom_query import PrometheusClient
    from train_types import WORKLOAD_FEATURES, SYSTEM_FEATURES, CATEGORICAL_LABEL_TO_VOCAB, NODE_STAT_POWER_LABEL
    import pandas as pd
    import numpy as np
    item = dict()

    for f in WORKLOAD_FEATURES:
        item[f] = random()
    for f in SYSTEM_FEATURES:
        if f in CATEGORICAL_LABEL_TO_VOCAB:
            item[f] = CATEGORICAL_LABEL_TO_VOCAB[f][0]
        else:
            item[f] = random()
    for l in NODE_STAT_POWER_LABEL:
        item[l] = random()*1000

    data_items = [item]
    node_stat_data = pd.DataFrame(data_items)
    print(node_stat_data.head())
    for f in WORKLOAD_FEATURES:
        node_stat_data[f] = node_stat_data[f].astype(np.float32)
    for f in SYSTEM_FEATURES:
        if f not in CATEGORICAL_LABEL_TO_VOCAB:
            node_stat_data[f] = node_stat_data[f].astype(np.float32)

    node_stat_data = pd.concat([node_stat_data]*10, ignore_index=True)
    prom_client = PrometheusClient()
    prom_client.latest_query_result[NODE_STAT_QUERY] = node_stat_data
    pipeline.train(prom_client)
    results = pipeline.predict(node_stat_data)
    print(results)