import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.class_prior_probs = {}
        self.feature_likelihoods = {}
        self.class_labels = []

    def fit(self, features, labels):
        self.class_labels = np.unique(labels)
        n_samples, n_features = features.shape

        for label in self.class_labels:
            features_for_class = features[labels == label]
            self.class_prior_probs[label] = len(features_for_class) / n_samples
            self.feature_likelihoods[label] = {}
            for feature_index in range(n_features):
                feature_values, counts = np.unique(features_for_class.iloc[:, feature_index], return_counts=True)
                self.feature_likelihoods[label][feature_index] = {value: count / len(features_for_class) for value, count in zip(feature_values, counts)}

    def predict(self, features):
        predictions = []
        for _, sample in features.iterrows():
            posterior_probs = {}
            for label in self.class_labels:
                prior_prob = self.class_prior_probs[label]
                likelihood = 1
                for feature_index in range(features.shape[1]):
                    likelihood *= self.feature_likelihoods[label].get(feature_index, {}).get(sample[feature_index], 1e-10)
                posterior_probs[label] = prior_prob * likelihood
            
            predicted_label = max(posterior_probs, key=posterior_probs.get)
            predictions.append(predicted_label)
        return predictions

# Example usage
data = {
    'feature1': [1, 1, 1, 0, 0, 0, 1, 0],
    'feature2': [1, 0, 1, 1, 0, 0, 0, 0],
    'label': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'B']
}

# df = pd.DataFrame(data)
# X = df[['feature1', 'feature2']]
# y = df['label']

# # Initialize and train the model
# model = NaiveBayesClassifier()
# model.fit(X, y)

# # Predicting on the training data
# predictions = model.predict(X)
# print(predictions)

# # Function to calculate accuracy
# def calculate_accuracy(true_labels, predicted_labels):
#     return np.sum(true_labels == predicted_labels) / len(true_labels)

# # Calculate accuracy
# accuracy_score = calculate_accuracy(y, predictions)
# print(f'Accuracy: {accuracy_score * 100:.2f}%')
