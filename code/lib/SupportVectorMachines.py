import numpy as np

class SupportVectorMachine:

    def __init__(self, learning_rate=0.001, regularization_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, features, labels):
        n_samples, n_features = features.shape

        # Convert labels to -1 and 1
        labels_transformed = np.where(labels <= 0, -1, 1)

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, sample in enumerate(features):
                # Check if the sample is correctly classified
                condition = labels_transformed[idx] * (np.dot(sample, self.weights) - self.bias) >= 1
                if condition:
                    # Update weights if the sample is correctly classified
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights)
                else:
                    # Update weights and bias if the sample is misclassified
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights - np.dot(sample, labels_transformed[idx]))
                    self.bias -= self.learning_rate * labels_transformed[idx]

    def predict(self, features):
        # Calculate the linear approximation
        linear_approximation = np.dot(features, self.weights) - self.bias
        return np.sign(linear_approximation)


