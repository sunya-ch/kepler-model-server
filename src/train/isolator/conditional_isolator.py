class ConditionalIsolator(Isolator):
    def __init__(self, feature_cols, label_cols, isolation_trainers, representative_usage_metric, isolation_goodness_threshold=0.75, target_usage_threshold=None):
        self.isolation_trainers = isolation_trainers
        self.representative_usage_metric = representative_usage_metric # e.g., cpu_cycles
        self.target_usage_threshold = target_usage_threshold
        self.isolation_goodness_threshold = isolation_goodness_threshold
        super(ConditionalIsolator, self).__init__(feature_cols, label_cols)

    def get_target_containers(self, data):
        sum_stat_by_container = data.reset_index().groupby(columns=[container_id_colname]).sum()
        if self.target_usage_threshold is not None:
            # filter large enough target
            sum_stat_by_container = sum_stat_by_container > self.target_usage_threshold
        return pd.unique(sum_stat_by_container[container_id_colname])

    def isolate(self, data, *args):
        target_containers = self.get_target_containers()
        if len(target_containers) == 0:
            return None
        isolate_data_list = []
        for target_container_id in target_containers:
            indexed_data = data.set_index(container_indexes)
            target_container_data, conditional_data = exclude_target_container_usage(indexed_data, target_container_id)
            for label_col in self.label_cols:
                lowest_err = None
                best_target_powers = None
                for trainer in self.isolation_trainers:
                    target_powers, isolation_goodness, err_val = trainer.train_eval(data, target_container_data, conditional_data, self.feature_cols, label_col)
                    if isolation_goodness < self.isolation_goodness_threshold:
                        continue
                    if lowest_err is None or lowest_err > err_val:
                        lowest_err = err_val
                        best_target_powers = target_powers
                if best_target_powers is None:
                    # next target
                    continue
                target_container_data[label_col] = target_powers
            isolate_data_list += [target_container_data]
        return  pd.concat(isolate_data_list).sort_index()