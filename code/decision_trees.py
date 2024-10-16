import numpy as np


def compute_gini_impurity(data):
    total_instances = len(data)
    if total_instances == 0:
        return 0

    classes, counts = np.unique(data, return_counts=True)
    probabilities = counts / total_instances

    gini = 1 - np.sum(probabilities**2)
    return gini


def compute_entropy(data):
    total_instances = len(data)
    if total_instances == 0:
        return 0

    classes, counts = np.unique(data, return_counts=True)
    probabilities = counts / total_instances
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value


def compute_mse(data):
    if len(data) == 0:
        return 0
    mean_value = np.mean(data)
    return np.mean((data - mean_value) ** 2)


def calculate_information_gain_with_entropy(data, feature, target):
    total_entropy = compute_entropy(target)

    # Get unique values of the feature
    feature_values = np.unique(feature)

    # Calculate the weighted entropy after the split
    weighted_entropy = 0
    total_instances = len(target)

    for value in feature_values:
        subset_indices = np.where(feature == value)[0]
        subset = target[subset_indices]
        subset_entropy = compute_entropy(subset)
        weighted_entropy += (len(subset) / total_instances) * subset_entropy

    # Information Gain
    ig = total_entropy - weighted_entropy
    return ig


def calculate_information_gain_with_gini(data, feature, target):
    total_gini = compute_gini_impurity(target)

    # Get unique values of the feature
    feature_values = np.unique(feature)

    # Calculate the weighted Gini impurity after the split
    weighted_gini = 0
    total_instances = len(target)

    for value in feature_values:
        subset_indices = np.where(feature == value)[0]
        subset = target[subset_indices]
        subset_gini = compute_gini_impurity(subset)
        weighted_gini += (len(subset) / total_instances) * subset_gini

    ig = total_gini - weighted_gini
    return ig


def information_gain_mse(data, feature, target):
    total_mse = compute_mse(target)

    # Get unique values of the feature
    feature_values = np.unique(feature)

    # Calculate the weighted MSE after the split
    weighted_mse = 0
    total_instances = len(target)

    for value in feature_values:
        subset_indices = np.where(feature == value)[0]
        subset = target[subset_indices]
        subset_mse = compute_mse(subset)
        weighted_mse += (len(subset) / total_instances) * subset_mse

    # Information Gain based on MSE
    ig = total_mse - weighted_mse
    return ig


# Example dataset

data = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C']
feature_A = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # Feature A
classes = np.array(['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])  # Target class

gini_value = compute_gini_impurity(data)

ig_value = calculate_information_gain_with_entropy(classes, feature_A, classes)

ig_value_gini = calculate_information_gain_with_gini(
    classes, feature_A, classes
)

print(f"Information Gain: {ig_value:.4f}")
print(f"Gini Impurity: {gini_value:.4f}")
print(f"Information Gain (Gini): {ig_value_gini:.4f}")


# Example dataset
feature_A = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # Feature A
target = np.array([10, 12, 11, 20, 21, 19, 22, 25])  # Target variable
ig_value_mse = information_gain_mse(target, feature_A, target)
print(f"Information Gain (MSE): {ig_value_mse:.4f}")
