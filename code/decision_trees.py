import numpy as np


class DecisionTreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label (if leaf node)


def gini_index(y):
    classes, counts = np.unique(y, return_counts=True)
    gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)
    return gini


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(p * np.log2(p) for p in probs if p > 0)


def find_best_split(X, y, criterion='gini'):
    best_gini = float('inf')
    best_entropy = float('inf')
    best_split = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = X[:, feature_index] > threshold

            y_left, y_right = y[left_mask], y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            if criterion == 'gini':
                gini = (len(y_left) / len(y)) * gini_index(y_left) + (
                    len(y_right) / len(y)
                ) * gini_index(y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)
                    print()
                    print(
                        f"Feature Index: {feature_index} and , Threshold:{threshold}"
                    )
                    print(f"Best Split: {best_split}")

            elif criterion == 'entropy':
                entropy_value = (len(y_left) / len(y)) * entropy(y_left) + (
                    len(y_right) / len(y)
                ) * entropy(y_right)
                if entropy_value < best_entropy:
                    best_entropy = entropy_value
                    best_split = (feature_index, threshold)

    return best_split


def build_tree(X, y, criterion='gini', max_depth=None, depth=0):
    if len(set(y)) == 1:  # Only one class left
        return DecisionTreeNode(value=y[0])

    if max_depth is not None and depth >= max_depth:  # Maximum depth reached
        return DecisionTreeNode(
            value=np.bincount(y).argmax()
        )  # Majority class

    best_split = find_best_split(X, y, criterion)
    if best_split is None:  # No valid split found
        return DecisionTreeNode(value=np.bincount(y).argmax())

    feature_index, threshold = best_split
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold

    left_node = build_tree(
        X[left_mask], y[left_mask], criterion, max_depth, depth + 1
    )
    right_node = build_tree(
        X[right_mask], y[right_mask], criterion, max_depth, depth + 1
    )

    return DecisionTreeNode(feature_index, threshold, left_node, right_node)


def predict(node, x):
    if node.value is not None:  # Leaf node
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)


# Example usage
# tree = build_tree(X_train, y_train, criterion='gini', max_depth=5)
# prediction = predict(tree, new_data_point)


class ClassificatonInformationGain:

    def __init__(self, data, labels, loss_f):
        self.data = data
        self.labels = labels
        self.loss_f = loss_f

        if self.total_instances == 0:
            return 0

    def compute_values(data, feature, target):
        total_loss = self.compute_loss(target)
        feature_values = np.unique(feature)
        weighted_entropy = 0
        total_instances = len(target)

        for value in feature_values:
            subset_indices = np.where(feature == value)[0]
            subset = target[subset_indices]
            subset_entropy = self.compute_loss(subset)
            weighted_entropy += (
                len(subset) / total_instances
            ) * subset_entropy

        ig = total_loss - weighted_entropy
        return ig

    def compute_loss(X):

        total_instances = len(X)
        classes, counts = np.unique(X, return_counts=True)
        probabilities = counts / total_instances

        if self.loss_f == "gini":
            return 1 - np.sum(probabilities**2)
        elif self.loss_f == "entropy":
            return -np.sum(probabilities * np.log2(probabilities))


class RegressionInformationGain:
    def __init__(self, data):
        self.data = data

    def compute_mse(self, data):
        if len(data) == 0:
            return 0
        mean_value = np.mean(data)
        return np.mean((data - mean_value) ** 2)

    def information_gain_mse(data, feature, target):
        total_mse = self.compute_mse(target)
        feature_values = np.unique(feature)
        weighted_mse = 0
        total_instances = len(target)

        for value in feature_values:
            subset_indices = np.where(feature == value)[0]
            subset = target[subset_indices]
            subset_mse = self.compute_mse(subset)
            weighted_mse += (len(subset) / total_instances) * subset_mse

        ig = total_mse - weighted_mse
        return ig


# Example dataset
classification_data = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C']
classification_feature = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # Feature A

# Example dataset
regression_data = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C']
regression_feature = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # Feature A
