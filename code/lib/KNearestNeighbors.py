import numpy as np
from collections import Counter


def calculate_minkowski_distance(point_a, point_b, p=2):
    return np.sum(np.abs(point_a - point_b) ** p) ** (1 / p)


class KNearestNeighbors:
    def __init__(self, num_neighbors=3, distance_metric='euclidean'):
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric

    def fit(self, training_features, training_labels):
        if len(training_features) != len(training_labels):
            raise ValueError(
                "Number of training samples must match number of labels."
            )
        self.training_features = training_features
        self.training_labels = training_labels

    def predict(self, test_features):
        if len(test_features) == 0:
            return []
        predictions = [
            self.get_most_common_label(test_instance)
            for test_instance in test_features
        ]
        return predictions

    def get_most_common_label(self, test_instance):
        # Compute distances from the test instance to all training instances
        distances = [
            self.calculate_distance(test_instance, train_instance)
            for train_instance in self.training_features
        ]

        # Get indices of the closest k neighbors
        k_nearest_indices = np.argsort(distances)[: self.num_neighbors]
        k_nearest_labels = [self.training_labels[i] for i in k_nearest_indices]

        # Majority vote to determine the predicted label
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_label

    def calculate_distance(self, point_a, point_b):
        if self.distance_metric == 'euclidean':
            return calculate_minkowski_distance(point_a, point_b, p=2)
        elif self.distance_metric == 'manhattan':
            return calculate_minkowski_distance(point_a, point_b, p=1)
        else:
            raise ValueError(
                f"Unsupported distance metric: {self.distance_metric}"
            )
