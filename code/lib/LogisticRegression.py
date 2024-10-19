import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.001, num_iterations=1000, schedule_type=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.schedule_type = schedule_type
        self.coefficients = None
        self.intercept = None

    def fit(self, features, target, method="gd"):
        num_samples, num_features = features.shape
        self.coefficients = np.zeros(num_features)
        self.intercept = 0

        if method == "gd":
            for iteration in range(self.num_iterations):
                linear_combination = np.dot(features, self.coefficients) + self.intercept
                probabilities = self.sigmoid_function(linear_combination)

                gradients_weights = (1 / num_samples) * np.dot(features.T, (probabilities - target))
                gradients_bias = (1 / num_samples) * np.sum(probabilities - target)

                self.coefficients -= self.learning_rate * gradients_weights
                self.intercept -= self.learning_rate * gradients_bias

                self.update_learning_rate(iteration)
                
        elif method == "sgd":
            for iteration in range(self.num_iterations):
                for i in range(num_samples):
                    linear_combination = np.dot(features[i], self.coefficients) + self.intercept
                    probability = self.sigmoid_function(linear_combination)

                    gradients_weights = (probability - target[i]) * features[i]
                    gradients_bias = probability - target[i]

                    self.coefficients -= self.learning_rate * gradients_weights
                    self.intercept -= self.learning_rate * gradients_bias

                self.update_learning_rate(iteration)

    def sigmoid_function(self, input_array):
        return 1 / (1 + np.exp(-input_array))

    def update_learning_rate(self, iteration):
        if self.schedule_type == "decay":
            self.learning_rate *= 1 / (1 + 0.1 * iteration)
        elif self.schedule_type == "step":
            if iteration % 100 == 0 and iteration > 0:
                self.learning_rate *= 0.5

    def predict(self, features):
        linear_combination = np.dot(features, self.coefficients) + self.intercept
        probabilities = self.sigmoid_function(linear_combination)
        class_predictions = [0 if prob <= 0.5 else 1 for prob in probabilities]
        return class_predictions
