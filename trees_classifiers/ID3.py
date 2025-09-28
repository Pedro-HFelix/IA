import numpy as np

class DecisionTreeID3:
    def __init__(self, features, target_class, data):
        self.features = features
        self.target_class = target_class
        self.data = data
        self.tree = None
        self.default_prediction = None

    def _calculate_entropy(self, data):
        total_samples = len(data)
        if total_samples == 0:
            return 0
        
        class_counts = data[self.target_class].value_counts()
        entropy = 0
        
        for count in class_counts:
            probability = count / total_samples
            entropy -= probability * np.log2(probability)
            
        return entropy
    
    def _calculate_information_gain(self, data, feature):
        total_samples = len(data)
        if total_samples == 0:
            return 0
            
        parent_entropy = self._calculate_entropy(data)
        
        unique_values = data[feature].unique()
        weighted_entropy = 0
        
        for value in unique_values:
            subset = data[data[feature] == value]
            probability = len(subset) / total_samples
            subset_entropy = self._calculate_entropy(subset)
            weighted_entropy += probability * subset_entropy
            
        information_gain = parent_entropy - weighted_entropy
        return information_gain
    
    def build_tree(self, data=None, features=None):
        if data is None:
            data = self.data
        if features is None:
            features = self.features.copy()

        if len(data[self.target_class].unique()) == 1:
            return data[self.target_class].iloc[0]

        if not features:
            return data[self.target_class].mode()[0]

        best_gain = -1
        best_feature = None
        min_branches = float('inf')
        for feature in features:
            gain = self._calculate_information_gain(data, feature)
            num_branches = data[feature].nunique()

            if gain > best_gain:
                best_gain = gain
                min_branches = num_branches
                best_feature = feature
            elif gain == best_gain:
                if num_branches < min_branches:
                    min_branches = num_branches
                    best_feature = feature


        if best_gain == 0:
            return data[self.target_class].mode()[0]

        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            
            if subset.empty:
                tree[best_feature][value] = data[self.target_class].mode()[0]
            else:
                remaining_features = [f for f in features if f != best_feature]
                tree[best_feature][value] = self.build_tree(subset, remaining_features)

        return tree

    def predict_sample(self, sample, tree):

        if not isinstance(tree, dict):
            return tree
        
        feature = next(iter(tree))
        value = sample.get(feature)

        if value not in tree[feature]:
            return self.default_prediction 
            
        return self.predict_sample(sample, tree[feature][value])

    def build(self):
        self.default_prediction = self.data[self.target_class].mode()[0] 
        self.tree = self.build_tree()
        return self

    def predict(self, sample):
        if self.tree is None:
            raise Exception("The model has not been trained yet")
        return self.predict_sample(sample, self.tree)
    
