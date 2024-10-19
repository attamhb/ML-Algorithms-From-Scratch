import numpy as np
from collections import Counter
from decision_trees import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self.bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self, features, labels):
        n_samples = features.shape[0]
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        return features[sample_indices], labels[sample_indices]

    def get_most_common_label(self, labels):
        counter = Counter(labels)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions_by_tree = np.swapaxes(all_predictions, 0, 1)
        final_predictions = np.array([self.get_most_common_label(pred) for pred in predictions_by_tree])
        return final_predictions
