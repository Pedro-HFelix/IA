import pandas as pd

class DecisionTreeCART:
    def __init__(self, data, features, target_class):
        self.data = data
        self.features = features
        self.target_class = target_class
        self.tree = None

    def build(self):
        processed_data = self._handle_missing_values(self.data.copy())
        self.tree = self._build_tree(processed_data, self.features.copy())
        return self

    def predict(self, sample):
        if self.tree is None:
            raise Exception("Model has not been trained yet")
        return self._traverse_tree(sample, self.tree)

    def _handle_missing_values(self, data):
        for feature in self.features:
            if pd.api.types.is_numeric_dtype(data[feature]):
                mean_val = data[feature].mean()
                data[feature] = data[feature].fillna(mean_val)
            else:
                mode_val = data[feature].mode()[0]
                data[feature] = data[feature].fillna(mode_val)
        return data
    
    def _calculate_gini(self, data):
        num_samples = len(data)
        if num_samples == 0: 
            return 0
        class_counts = data[self.target_class].value_counts()
        return 1 - sum((count / num_samples) ** 2 for count in class_counts)

    def _find_best_split_for_feature(self, data, feature):
        num_samples = len(data)
        best_gini = 1
        best_split = None

        if pd.api.types.is_numeric_dtype(data[feature]):
            unique_values = sorted(data[feature].unique())
            threshold_candidates = [(unique_values[i] + unique_values[i+1]) / 2 
                                    for i in range(len(unique_values)-1)]
            for threshold in threshold_candidates:
                left = data[data[feature] <= threshold]
                right = data[data[feature] > threshold]
                if len(left) == 0 or len(right) == 0: 
                    continue
                split_gini = (len(left)/num_samples) * self._calculate_gini(left) + \
                             (len(right)/num_samples) * self._calculate_gini(right)
                if split_gini < best_gini:
                    best_gini = split_gini
                    best_split = ("numeric", threshold)

        else:  
            unique_values = data[feature].unique()
            for value in unique_values:
                left = data[data[feature] == value]
                right = data[data[feature] != value]
                if len(left) == 0 or len(right) == 0: 
                    continue
                split_gini = (len(left)/num_samples) * self._calculate_gini(left) + \
                             (len(right)/num_samples) * self._calculate_gini(right)
                if split_gini < best_gini:
                    best_gini = split_gini
                    best_split = ("categorical", value)

        return best_gini, best_split

    def _build_tree(self, data, features):
        if len(data[self.target_class].unique()) == 1:
            return data[self.target_class].iloc[0]

        if not features:
            return data[self.target_class].mode()[0]

        best_overall_gini = 1
        best_feature, best_split = None, None

        for feature in features:
            split_gini, split_details = self._find_best_split_for_feature(data, feature)
            if split_gini is not None and split_gini < best_overall_gini:
                best_overall_gini = split_gini
                best_feature = feature
                best_split = split_details

        if best_feature is None or best_overall_gini == self._calculate_gini(data):
            return data[self.target_class].mode()[0]

        split_type, split_value = best_split
        tree = {best_feature: {}}

        remaining_features = [f for f in features if f != best_feature]

        if split_type == "numeric":
            left_subset = data[data[best_feature] <= split_value]
            right_subset = data[data[best_feature] > split_value]
            tree[best_feature][("<= ", split_value)] = self._build_tree(left_subset, features)
            tree[best_feature][("> ", split_value)] = self._build_tree(right_subset, features)
        else:  
            left_subset = data[data[best_feature] == split_value]
            right_subset = data[data[best_feature] != split_value]
            tree[best_feature][("==", split_value)] = self._build_tree(left_subset, features)
            tree[best_feature][("!=", split_value)] = self._build_tree(right_subset, features)

        return tree

    def _traverse_tree(self, sample, node):
        if not isinstance(node, dict):
            return node

        feature = next(iter(node))
        value_from_sample = sample.get(feature)

        for (op, val), subtree in node[feature].items():
            if op == "<=" and value_from_sample <= val:
                return self._traverse_tree(sample, subtree)
            elif op == ">" and value_from_sample > val:
                return self._traverse_tree(sample, subtree)
            elif op == "==" and value_from_sample == val:
                return self._traverse_tree(sample, subtree)
            elif op == "!=" and value_from_sample != val:
                return self._traverse_tree(sample, subtree)

        return self.data[self.target_class].mode()[0]

