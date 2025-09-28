import numpy as np
import pandas as pd

class DecisionTreeC45:
    def __init__(self, data, features, target_class):
        self.data = data
        self.features = features
        self.target_class = target_class
        self.tree = None
        self.default_prediction = None

    def build(self):
        self.default_prediction = self.data[self.target_class].mode()[0]
        
        processed_data = self._handle_missing_values(self.data.copy())
        
        self.tree = self._build_tree(processed_data, self.features.copy())
        return self

    def predict(self, sample):
        if self.tree is None:
            raise Exception("The model has not been trained yet")
        return self._traverse_tree(sample, self.tree)

    def _handle_missing_values(self, data):
        for col in self.features:
            if pd.api.types.is_numeric_dtype(data[col]):
                mean_val = data[col].mean()
                data[col][:] = data[col].fillna(mean_val)
            else:
                mode_val = data[col].mode()[0]
                data[col][:] = data[col].fillna(mode_val)
        return data

    def _calculate_entropy(self, data):
        num_samples = len(data)
        if num_samples == 0: return 0
        class_counts = data[self.target_class].value_counts()
        entropy = 0
        for count in class_counts:
            prob = count / num_samples
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy

    def _calculate_split_info(self, data, feature, threshold=None):
        num_samples = len(data)
        split_info = 0
        if threshold is not None:  
            groups = [data[data[feature] <= threshold], data[data[feature] > threshold]]
        else: 
            groups = [data[data[feature] == v] for v in data[feature].unique()]
        for g in groups:
            if len(g) > 0:
                proportion = len(g) / num_samples
                if proportion > 0:
                    split_info -= proportion * np.log2(proportion)
        return split_info

    def _calculate_gain_ratio(self, data, feature):
        parent_entropy = self._calculate_entropy(data)
        num_samples = len(data)
        if num_samples == 0: return 0, None
        if pd.api.types.is_numeric_dtype(data[feature]):
            unique_values = sorted(data[feature].unique())
            best_gain_ratio = -1
            best_threshold = None
            if len(unique_values) < 2: return 0, None
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                subset_le = data[data[feature] <= threshold]
                subset_gt = data[data[feature] > threshold]
                weighted_entropy = ((len(subset_le) / num_samples * self._calculate_entropy(subset_le)) + (len(subset_gt) / num_samples * self._calculate_entropy(subset_gt)))
                info_gain = parent_entropy - weighted_entropy
                split_info = self._calculate_split_info(data, feature, threshold)
                gain_ratio = info_gain / split_info if split_info != 0 else 0
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_threshold = threshold
            return best_gain_ratio, best_threshold
        else:
            unique_values = data[feature].unique()
            weighted_entropy = 0
            for value in unique_values:
                subset = data[data[feature] == value]
                proportion = len(subset) / num_samples
                weighted_entropy += proportion * self._calculate_entropy(subset)
            info_gain = parent_entropy - weighted_entropy
            split_info = self._calculate_split_info(data, feature)
            gain_ratio = info_gain / split_info if split_info != 0 else 0
            return gain_ratio, None

    def _build_tree(self, data, features):
        if len(data[self.target_class].unique()) == 1:
            return data[self.target_class].iloc[0]
        if not features:
            return data[self.target_class].mode()[0]
        best_gain_ratio = -1
        best_feature, best_threshold = None, None
        min_branches = float('inf')
        for feature in features:
            gain_ratio, threshold = self._calculate_gain_ratio(data, feature)
            num_branches = 2 if threshold is not None else data[feature].nunique()
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                min_branches = num_branches
                best_feature = feature
                best_threshold = threshold
            elif gain_ratio == best_gain_ratio:
                if num_branches < min_branches:
                    min_branches = num_branches
                    best_feature = feature
                    best_threshold = threshold
        if best_gain_ratio <= 0: return data[self.target_class].mode()[0]
        tree = {best_feature: {}}
        if best_threshold is not None:
            subset_le = data[data[best_feature] <= best_threshold]
            subset_gt = data[data[best_feature] > best_threshold]
            tree[best_feature][f"<= {best_threshold}"] = self._build_tree(subset_le, features)
            tree[best_feature][f"> {best_threshold}"] = self._build_tree(subset_gt, features)
        else:
            remaining_features = [f for f in features if f != best_feature]
            for value in data[best_feature].unique():
                subset = data[data[best_feature] == value]
                if subset.empty:
                    tree[best_feature][value] = self.default_prediction
                else:
                    tree[best_feature][value] = self._build_tree(subset, remaining_features)
        return tree

    def _traverse_tree(self, sample, node):
        if not isinstance(node, dict):
            return node
        feature = next(iter(node))
        value = sample.get(feature)
        if value is None:
            return self.default_prediction
        for key, subtree in node[feature].items():
            if "<=" in str(key) or ">" in str(key):
                limit = float(str(key).split()[1])
                branch = f"<= {limit}" if value <= limit else f"> {limit}"
                if branch == key:
                    return self._traverse_tree(sample, subtree)
            else:
                if value == key:
                    return self._traverse_tree(sample, subtree)
        return self.default_prediction
    