import numpy as np
from collections import Counter


class DecisionTreeNode:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        *,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self,
        min_samples_split=2,
        max_depth=100,
        n_features=None,
        criterion='gini',
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.criterion = criterion

    def fit(self, X, y):
        self.n_features = (
            X.shape[1]
            if self.n_features is None
            else min(X.shape[1], self.n_features)
        )
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self.get_most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        feature_indices = np.random.choice(
            n_feats, self.n_features, replace=False
        )

        best_feature, best_threshold = self.find_optimal_split(
            X, y, feature_indices
        )

        left_indices, right_indices = self.split(
            X[:, best_feature], best_threshold
        )
        left_child = self.build_tree(
            X[left_indices, :], y[left_indices], depth + 1
        )
        right_child = self.build_tree(
            X[right_indices, :], y[right_indices], depth + 1
        )
        return DecisionTreeNode(
            best_feature, best_threshold, left_child, right_child
        )

    def find_optimal_split(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self.compute_information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def compute_information_gain(self, y, X_column, threshold):
        if self.criterion == 'entropy':
            parent_measure = self.compute_entropy(y)
            left_indices, right_indices = self.split(X_column, threshold)

            if len(left_indices) == 0 or len(right_indices) == 0:
                return 0

            child_measure = (
                len(left_indices) / len(y)
            ) * self.compute_entropy(y[left_indices]) + (
                len(right_indices) / len(y)
            ) * self.compute_entropy(
                y[right_indices]
            )
        elif self.criterion == 'gini':
            parent_measure = self.compute_gini_index(y)
            left_indices, right_indices = self.split(X_column, threshold)

            if len(left_indices) == 0 or len(right_indices) == 0:
                return 0

            child_measure = (
                len(left_indices) / len(y)
            ) * self.compute_gini_index(y[left_indices]) + (
                len(right_indices) / len(y)
            ) * self.compute_gini_index(
                y[right_indices]
            )
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'.")

        return parent_measure - child_measure

    def split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def compute_entropy(self, y):
        histogram = np.bincount(y)
        probabilities = histogram / len(y)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def compute_gini_index(self, y):
        histogram = np.bincount(y)
        probabilities = histogram / len(y)
        return 1 - np.sum([p ** 2 for p in probabilities])

    def get_most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left_child)
        return self.traverse_tree(x, node.right_child)


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=None):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def variance(self, y):
        pass

    def compute_information_gain(self, y, X_column, threshold):
        pass

    def split(self, X_column, threshold):
        pass

    def compute_mse(self, y_true, y_pred):
        pass

    def get_best_split(self, X, y):
        pass

    def build_tree(self, X, y, depth=0):
        pass

    def traverse_tree(self, x, node):
        pass

    def prune_tree(self):
        pass
