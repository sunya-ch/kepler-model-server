
<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [Promethes Query](#promethes-query)
- [Extractor](#extractor)
- [Isolator](#isolator)
- [Trainer](#trainer)
- [Pipeline](#pipeline)
- [Estimator Power Prediction](#estimator-power-prediction)
- [Estimator Model Request (to Model Server)](#estimator-model-request-to-model-server)
- [Estimator Power Request (from Collector)](#estimator-power-request-from-collector)

<!-- /TOC -->

# Available Tests

## Promethes Query
The test is for testing making query to prometheus server. The output will be saved in json format.

Requirements:
- kepler-exporter running and export metrics to prometheus server
- Prometheus server serves at localhost:9090

Run:
```bash
python prom_test.py
```

Reuse:
```python
from prom_test import process as prom_process
prom_process()
```

Optional arguments:
 - server: prometheus server address (default: localhost:9090)
 - save_path: path to save a query response (default: data/prom_output)
 - save_name: response filename to save (default: prom_response.json)
 - interval: query interval (default: 300s)
 - step: query step (default: 3s)

## Extractor
The test is for testing extractors from saved prometheus response. The output will be saved in csv format for node-level and container-level data of each feature group.

Requirements:
- prometheus response in json (default: data/prom_output/prom_repsonse.json)

Run:
```bash
python extractor_test.py
```

Reuse:
```python
from extractor_test import process as extractor_process
extractor_process(query_results, feature_group)
```

Optional arguments:
 - save_path: path to save a extracted result (default: data/extractor_output)
 - customize_extractors: additional extractor names (default: [])
 - energy_source: target energy source (default: rapl)
 - num_of_unit: number of component units (default: 2)

## Isolator
The test is for testing isolators from container-level extracted data. The output will be saved in csv format for container-level data of each feature group for each extractor.

Requirements:
- extractor output (default: data/extractor_output)

Run:
```bash
python isolator_test.py
```

Reuse:
```python
from isolator_test import process as isolate_process
isolate_process()
```
Optional arguments:
 - save_path: path to save a isolated result (default: data/isolator_output)
 - extract_path: input path of extracted result (default: data/extractor_output)
 - test_isolators: list of basic isolator instances (default: [MinIsolator, NoneIsolator])
 - customize_isolators: additional isolator instances (default: [])

## Trainer
The test is for testing trainers for node-level power from extracted data and for container-level power from isolated data. The trained models will be saved in model path (default: src/models) and accuracy results will be printed.

Requirements:
- extractor output (default: data/extractor_output)
- isolator output (default: data/isolator_output)

Run
```bash
python trainer_test.py
```

Reuse:
```python
from isolator_test import get_isolate_results
from extractor_test import get_extract_results
from trainer_test import process as trainer_process

# node-level, use extractor_output at extractor_output_path (default: data/extractor_output)
extractor_results = get_extract_results(extractor_name, node_level=True, save_path=extractor_output_path)
for feature_group, result in extractor_results.items():
    trainer_process(True, feature_group, result)

# container-level, use isolator_output at isolator_output_path (default: data/isolator_output)
isolator_results = get_isolate_results(isolator_name, extractor_name, save_path=isolator_output_path)
for feature_group, result in isolator_results.items():
    trainer_process(False, feature_group, result)
```
Optional arguments:
 - trainer_names: trainer names to be initiated (default: test_trainer_names)
 - energy_source: target energy source (default: rapl)
 - power_columns: extracted power columns (default: generated power columns from prom_response)
 - pipeline_name: pipeline name to be saved under model top path, /models, (default: default)

## Pipeline
The test is for testing pipeline execution which is composed of testing extractor, testing isolator, and testing trainer. The input is saved prometheus response. The trained models will be saved in model path (default: src/models)/

Requirements:
- prometheus response in json (default: data/prom_output/prom_repsonse.json)

Run:
```bash
python pipeline_test.py
```

Reuse:
```python
from pipeline_test import process as pipeline_process
pipeline_process()
```

Optional arguments:
 - prom_save_path: path of prometheus response (default: data/prom_output)
 - prom_save_name: prometheus response filename (default: prom_response)
 - target_energy_sources: list of target energy sources (default: [rapl])
 - extractors: list of extractor instances (default: [DefaultExtractor])
 - isolators: list of isolator instances (default: [MinIsolator, NoneIsolator])
 - abs_trainer_names: list of trainer names to learn for DynPower (default: test_trainer_names)
 - dyn_trainer_names: list of trainer names to learn for AbsPower (default: test_trainer_names)

## Estimator Power Prediction
 The test is for testing applying a trained model to extracted and isolated data for node-level and container-level power, respectively. Prediction error will be printed. 

 Requirements:
 - trained models (default: ../src/models/default)

 Run:
 ```
 python estimate_mode_test.py
 ```

 Reuse (test_model):
 ```python
 from util import ModelOutputType, model_toppath, DEFAULT_PIPELINE
 from isolator_test import get_isolate_results
 from extractor_test import get_extract_results, get_expected_power_columns

 from estimator_model_test import test_model
 
 # model folder under model_toppath which is ../src/models by default
 pipeline_name = DEFAULT_PIPELINE
 # get_expected_power_columns(energy_components=test_energy_components, num_of_unit=test_num_of_unit)
 power_columns = get_expected_power_columns()

 # node-level, use extractor_output at extractor_output_path (default: data/extractor_output)
 extractor_results = get_extract_results(extractor_name, node_level=True, save_path=extractor_output_path)
 for feature_group, result in extractor_results.items():
    group_path = get_model_group_path(model_toppath, ModelOutputType.AbsPower, feature_group, energy_source, pipeline_name)
    test_model(group_path, model_name, result, power_columns)

 # container-level, use isolator_output at isolator_output_path (default: data/isolator_output)
 isolator_results = get_isolate_results(isolator_name, extractor_name, save_path=isolator_output_path)
 for feature_group, result in isolator_results.items():
    group_path = get_model_group_path(model_toppath, ModelOutputType.DynPower, feature_group, energy_source, pipeline_name)
    test_model(group_path, model_name, result, power_columns)
 ```
Optional arguments:
 - power_range: power range to compute scaled error in percentage

## Estimator Model Request (to Model Server)
The test is for testing estimator as a client to model server by making a request for best model for a specific feature group and performing a prediction by the loaded model. Output of prediction will be printed. 

Requirements:
- trained model
- server running

    ```bash
    python ../src/server/model_server.py
    ```

Run:
```bash
python estimator_model_request_test.py
```

## Estimator Power Request (from Collector)
The test if for testing estimator as a server to serve a power request from Kepler collector. Response output will be printed

Requirements:
- model initial url defined in [util/loader.py](../src/util/loader.py) is available.
- estimator running

    ```bash
    python ../src/estimate/estimator.py
    ```

Run:
```bash
python estimator_power_request_test.py
```