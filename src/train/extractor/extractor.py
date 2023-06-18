import os
import sys
import pandas as pd
from abc import ABCMeta, abstractmethod

util_path = os.path.join(os.path.dirname(__file__), '..', '..', 'util')
sys.path.append(util_path)

from train_types import FeatureGroups, FeatureGroup, SYSTEM_FEATURES
from prom_types import TIMESTAMP_COL, SOURCE_COL, get_energy_unit, \
    usage_ratio_query,node_info_query, \
        energy_component_to_query, feature_to_query, \
            pkg_id_column, container_id_cols, node_info_column
from extract_types import container_id_colname, ratio_to_col, component_to_col, UNKNOWN_NODE_INFO, get_unit_vals, container_level_index, node_level_index
from preprocess import correct_missing_metric_to_watt, drop_zero_column, find_correlations

# append ratio for each unit
def append_ratio_for_pkg(feature_power_data, is_aggr, query_results, power_columns):
    unit_vals = get_unit_vals(power_columns)
    if len(unit_vals) == 0:
        # not relate/not append
        return
    use_default_ratio = False
    default_ratio = 1/len(unit_vals)
    if usage_ratio_query not in query_results:
        use_default_ratio = True
    else:
        ratio_df = query_results[usage_ratio_query]
        if is_aggr:
            ratio_df = ratio_df.groupby([TIMESTAMP_COL, pkg_id_column]).sum()[usage_ratio_query]
        else:
            ratio_df[container_id_colname] = ratio_df[container_id_cols].apply(lambda x: '/'.join(x), axis=1)
            ratio_df = ratio_df.groupby([TIMESTAMP_COL, pkg_id_column, container_id_colname]).sum()[usage_ratio_query]   
    ratio_colnames = []
    for unit_val in unit_vals:
        ratio_colname = ratio_to_col(unit_val)
        if use_default_ratio:
            feature_power_data[ratio_colname] = default_ratio
        else:
            target_ratio_df = ratio_df.xs(unit_val, level=1)
            feature_power_data = feature_power_data.join(target_ratio_df).dropna()
            feature_power_data = feature_power_data.rename(columns={usage_ratio_query: ratio_colname})
        ratio_colnames +=[ratio_colname]
    tmp_total_col = 'total_ratio'
    feature_power_data[tmp_total_col] = feature_power_data[ratio_colnames].sum(axis=1)
    for ratio_colname in ratio_colnames:
        feature_power_data[ratio_colname] /= feature_power_data[tmp_total_col]
    return feature_power_data.drop(columns=[tmp_total_col])

class Extractor(metaclass=ABCMeta):
    # isolation abstract: should return dataFrame of features and labels
    @abstractmethod
    def extract(self, query_results, feature_group):
        return NotImplemented

# extract data from query 
# for node-level
# return DataFrame (index=timestamp, column=[features][power columns][node_type]), power_columns

class DefaultExtractor(Extractor):

    def get_workload_feature_data(self, query_results, features):
        feature_data = None
        container_df_map = dict()
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                return None
            aggr_query_data = query_results[query].copy()
            aggr_query_data.rename(columns={query: feature}, inplace=True)
            aggr_query_data[container_id_colname] = aggr_query_data[container_id_cols].apply(lambda x: '/'.join(x), axis=1)
            # separate for each container_id
            container_id_list = pd.unique(aggr_query_data[container_id_colname])
           
            df = pd.DataFrame()
            for container_id in container_id_list:
                container_df = aggr_query_data[aggr_query_data[container_id_colname]==container_id]
                container_df.set_index([TIMESTAMP_COL], inplace=True)
                if len(container_df) > 1:
                    # find current value from aggregated query, dropna remove the first value
                    df = container_df.sort_index()[[feature]].diff().dropna()
                else:
                    df = container_df.sort_index()[[feature]]
                # if delta < 0, set to 0 (unexpected)
                df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                if container_id in container_df_map:
                    # previously found container
                    container_df_map[container_id] = pd.concat([container_df_map[container_id], df], axis=1)
                else:
                    # newly found container
                    container_df_map[container_id] = df
        container_df_list = []
        for container_id, container_df in container_df_map.items():
            container_df[container_id_colname] = container_id
            container_df_list += [container_df]
        feature_data = pd.concat(container_df_list)
        # fill empty timestamp
        feature_data.fillna(0, inplace=True)
        # return with reset index for later aggregation
        return feature_data.reset_index()

    def get_system_feature_data(self, query_results, features):
        feature_data_list = []
        for feature in features:
            query = feature_to_query(feature)
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                return None
            aggr_query_data = query_results[query].copy()
            aggr_query_data.rename(columns={query: feature}, inplace=True)
            aggr_query_data = aggr_query_data.groupby([TIMESTAMP_COL]).mean().sort_index()
            feature_data_list += [aggr_query_data]
        feature_data = pd.concat(feature_data_list, axis=1).astype(int)
        return feature_data

    # return with timestamp index
    def get_power_data(self, query_results, energy_components, source):
        power_data_list = []
        for component in energy_components:
            unit_col = get_energy_unit(component)
            query = energy_component_to_query(component)
            if query not in query_results:
                print(query, 'not in', query_results)
                return None
            aggr_query_data = query_results[query].copy()
            # filter source
            aggr_query_data = aggr_query_data[aggr_query_data[SOURCE_COL] == source]
            if unit_col is not None:
                # sum over mode
                aggr_query_data = aggr_query_data.groupby([unit_col, TIMESTAMP_COL]).sum().reset_index().set_index(TIMESTAMP_COL)
                # add per unit_col
                unit_vals = pd.unique(aggr_query_data[unit_col])
                for unit_val in unit_vals:
                    df = aggr_query_data[aggr_query_data[unit_col]==unit_val].copy()
                    # rename
                    colname = component_to_col(component, unit_col, unit_val)
                    df.rename(columns={query: colname}, inplace=True)
                    # find current value from aggregated query
                    df = df.sort_index()[colname].diff().dropna()
                    df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                    power_data_list += [df]
            else:
                # sum over mode
                aggr_query_data = aggr_query_data.groupby([TIMESTAMP_COL]).sum()
                # rename
                colname = component_to_col(component)
                aggr_query_data.rename(columns={query: colname}, inplace=True)
                # find current value from aggregated query
                df = aggr_query_data.sort_index()[colname].diff().dropna()
                df = df.mask(df.lt(0)).ffill().fillna(0).convert_dtypes()
                power_data_list += [df]
        power_data = pd.concat(power_data_list, axis=1).dropna()
        return power_data

    def get_system_category(self, query_results):
        node_info_data = None
        if node_info_query in query_results:
            node_info_data = query_results[node_info_query][[TIMESTAMP_COL, node_info_query]].set_index(TIMESTAMP_COL)
            node_info_data.rename(columns={node_info_query: node_info_column}, inplace=True)
        return node_info_data

    def extract(self, query_results, energy_components, feature_group, energy_source, node_level, aggr=True):
        power_data = self.get_power_data(query_results, energy_components, energy_source)
        if power_data is None:
            return None, None, None
        power_data = drop_zero_column(power_data, power_data.columns)
        power_columns = power_data.columns
        features = FeatureGroups[FeatureGroup[feature_group]]
        workload_features = [feature for feature in features if feature not in SYSTEM_FEATURES]
        system_features = [feature for feature in features if feature in SYSTEM_FEATURES]
        feature_data = self.get_workload_feature_data(query_results, workload_features)
        if feature_data is None:
            return None, None, None
        feature_power_data = correct_missing_metric_to_watt(feature_data, power_data, workload_features, power_columns)
        feature_data = drop_zero_column(feature_data, workload_features)
        is_aggr = node_level and aggr
        if is_aggr:
            # sum stat of all containers
            sum_feature = feature_power_data.groupby([TIMESTAMP_COL]).sum()[workload_features]
            mean_power = feature_power_data.groupby([TIMESTAMP_COL]).mean()[power_columns]
            feature_power_data = sum_feature.join(mean_power)
        else:
            feature_power_data = feature_power_data.groupby([TIMESTAMP_COL, container_id_colname]).sum()
        if len(system_features) > 0:
            system_feature_data = self.get_system_feature_data(query_results, system_features)
            feature_power_data = feature_power_data.join(system_feature_data).sort_index().dropna()
        else: 
            feature_power_data = feature_power_data
        feature_power_data = feature_power_data.sort_index()
        node_info_data = self.get_system_category(query_results)
        if node_info_data is None:
            feature_power_data[node_info_column] = UNKNOWN_NODE_INFO
        else:
            feature_power_data = feature_power_data.join(node_info_data)
        corr = find_correlations(energy_source, feature_power_data, power_columns, workload_features)
        feature_power_data = append_ratio_for_pkg(feature_power_data, is_aggr, query_results, power_columns)
        return  feature_power_data, power_columns, corr
    
    def get_node_types(self, query_results):
        node_info_data = self.get_system_category(query_results)
        if node_info_data is None:
            print("No Node Info")
            return None, None
        return pd.unique(node_info_data[node_info_column]), node_info_data

